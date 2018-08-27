import math
from event.io.ontology import OntologyLoader


class MentionDetector:
    def __init__(self, **kwargs):
        super().__init__()
        self.unknown_type = "UNKNOWN"

    def predict(self, *input):
        pass


class BaseRuleDetector(MentionDetector):
    def __init__(self, config, token_vocab):
        super().__init__()
        self.onto = OntologyLoader(config.ontology_path)
        self.token_vocab = token_vocab

    def predict(self, *input):
        words, _, l_feature, word_meta, sent_meta = input[0]
        center = math.floor(len(words) / 2)
        lemmas = [features[0] if features else '' for features in l_feature]
        words = self.token_vocab.reveal_origin(words)
        return self.predict_by_word(words, lemmas, l_feature, center)

    def predict_by_word(self, words, lemmas, l_feature, center):
        raise NotImplementedError('Please implement the rule detector.')


class MarkedDetector(BaseRuleDetector):
    """
    Assume there is one field already marked with event type.
    """

    def __init__(self, config, token_vocab, marked_field_index=-2):
        super().__init__(config, token_vocab)
        self.marked_field_index = marked_field_index

    def predict(self, *input):
        words, _, l_feature, word_meta, sent_meta = input[0]
        center = math.floor(len(words) / 2)
        return l_feature[center][self.marked_field_index]


class FrameMappingDetector(BaseRuleDetector):
    def __init__(self, config, token_vocab):
        super().__init__(config, token_vocab)
        self.lex_mapping = self.load_frame_lex(config.frame_lexicon)
        # These word list are old word list from seedling.
        # self.entities, self.events, self.relations = self.load_wordlist(
        #     config.entity_list, config.event_list, config.relation_list
        # )

        self.event_onto = self.onto.event_onto_text()


    def load_frame_lex(self, frame_path):
        import xml.etree.ElementTree as ET
        import os

        ns = {'berkeley': 'http://framenet.icsi.berkeley.edu'}

        lex_mapping = {}

        for file in os.listdir(frame_path):
            with open(os.path.join(frame_path, file)) as f:
                tree = ET.parse(f)
                frame = tree.getroot()
                frame_name = frame.get('name')

                lexemes = []
                for lexUnit in frame.findall('berkeley:lexUnit', ns):
                    lex = lexUnit.get('name')
                    lexeme = lexUnit.findall(
                        'berkeley:lexeme', ns)[0].get('name')
                    lexemes.append(lexeme)

                lex_text = ' '.join(lexemes)
                if lex_text not in lex_mapping:
                    lex_mapping[lex_text] = []

                lex_mapping[lex_text].append(frame_name)
        return lex_mapping

    def load_wordlist(self, entity_file, event_file, relation_file):
        events = {}
        entities = {}
        relations = {}
        with open(event_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, ontology = line.strip().split()
                    events[word] = ontology
        with open(entity_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, ontology = line.strip().split()
                    entities[word] = ontology
        with open(relation_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if parts:
                    event_type = parts[0]
                    args = parts[1:]

                    if event_type not in relations:
                        relations[event_type] = {}

                    for arg in args:
                        arg_role, arg_types = arg.split(":")
                        relations[event_type][arg_role] = arg_types.split(",")

        return entities, events, relations

    def predict(self, *input):
        words, _, l_feature, word_meta, sent_meta = input[0]

        center = math.floor(len(words) / 2)
        extracted_type = l_feature[center][-1]

        if extracted_type in self.event_onto:
            return extracted_type
        else:
            return None

    def predict_args(self, center, event_type, data):
        if event_type not in self.event_onto:
            return {}

        expected_args = self.event_onto[event_type]['args']

        filled_args = dict([(k, None) for k in expected_args])
        num_to_fill = len(filled_args)

        relation_lookup = {}
        for role, content in expected_args.items():
            for t in content['restrictions']:
                relation_lookup[t] = role

        words, _, l_feature, word_meta, sent_meta = data

        # lemmas = [features[0] if features else None for features in l_feature]
        # deps = [features[5] if features else None for features in l_feature]
        onto_types = [features[-1] if features else None for features in
                      l_feature]

        for distance in range(1, center + 1):
            left = center - distance
            right = center + distance

            left_type = onto_types[left]
            right_type = onto_types[right]

            # if left_lemma and left_type in self.entities:
            #     arg_type = self.check_arg(lemmas[center], event_type,
            #                               left_lemma, deps)
            if left_type in relation_lookup:
                possible_rel = relation_lookup[left_type]
                if filled_args[possible_rel] is None:
                    filled_args[possible_rel] = (left, left_type)
                    num_to_fill -= 1

            # if right_lemma and right_lemma in self.entities:
            #     arg_type = self.check_arg(context[center], event_type,
            #                               right_lemma, deps)
            if right_type in relation_lookup:
                possible_rel = relation_lookup[right_type]
                if filled_args[possible_rel] is None:
                    filled_args[possible_rel] = (right, right_type)
                    num_to_fill -= 1

            if num_to_fill == 0:
                break

        return filled_args

    def check_arg(self, predicate, event_type, arg_lemma, features):
        unknown_type = "O"

        entity_type = unknown_type
        if arg_lemma in self.entities:
            entity_type = self.entities[arg_lemma]

        if not entity_type == unknown_type:
            return entity_type

        return None

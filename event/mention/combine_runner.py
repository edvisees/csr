from event.io.csr import CSR
import logging
import json
from collections import defaultdict
import glob
import os

from traitlets import (
    Unicode,
    Bool,
)
from traitlets.config.loader import PyFileConfigLoader
from event.mention.params import DetectionParams
from event.util import find_by_id
from event.io.readers import (
    ConllUReader,
    Vocab,
)
from event.mention.detection_runners import DetectionRunner
from event.io.ontology import (
    OntologyLoader,
    MappingLoader,
)
from event import resources
import nltk


def add_entity_relations(relation_file, edl_entities, csr):
    with open(relation_file) as rf:
        data = json.load(rf)

        for relation in data:
            for rel in relation['rels']:
                args = []
                for arg_name, relen in rel.items():
                    if arg_name == 'rel':
                        # Not an argument field.
                        continue

                    if relen not in edl_entities:
                        logging.error(
                            "Relation entities [{}] not found at {}".format(
                                relen, relation_file)
                        )
                        continue

                    en_id = edl_entities[relen].id
                    args.append((arg_name, en_id))

                if len(args) < 2:
                    logging.error(
                        "Insufficent number of fields {} in relation "
                        "at {}. some might be filtered".format(
                            len(rel), relation_file))
                    continue

                csr.add_relation(
                    'aida', args, rel['rel'], 'opera.relations.xiang'
                )


def add_edl_entities(edl_file, csr):
    edl_component_id = 'opera.entities.edl'

    edl_entities = {}

    if not edl_file:
        return
    with open(edl_file) as edl:
        data = json.load(edl)

        for entity_sent in data:
            docid = entity_sent['docID']
            csr.add_doc(docid, 'text', 'en')

            for entity in entity_sent['namedMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                head_span = entity['head_span']

                ent = csr.add_entity_mention(
                    head_span, mention_span, entity['mention'], 'aida',
                    entity['type'], entity_form='named',
                    component=edl_component_id)

                if ent:
                    edl_entities[entity['@id']] = ent

            for entity in entity_sent['nominalMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                head_span = entity['head_span']

                ent = csr.add_entity_mention(
                    head_span, mention_span, entity['mention'], 'aida',
                    entity['type'], entity_form='nominal',
                    component=edl_component_id)

                if ent:
                    edl_entities[entity['@id']] = ent

    return edl_entities


def recover_via_token(tokens, token_ids):
    if not token_ids:
        return None

    first_token = tokens[token_ids[0]]
    last_token = tokens[token_ids[-1]]

    text = ""
    last_end = -1
    for tid in token_ids:
        token_span, token_text = tokens[tid]

        if not last_end == -1 and token_span[0] - last_end > 0:
            gap = token_span[0] - last_end
        else:
            gap = 0
        last_end = token_span[1]

        text += ' ' * gap
        text += token_text

    return (first_token[0][0], last_token[0][1]), text


def fix_event_type_from_frame(origin_onto, origin_type, frame_type):
    if 'Contact' in origin_type:
        if frame_type == 'Quantity':
            return 'framenet', frame_type
    if 'Movement_Transportperson' == origin_type:
        if frame_type == 'Arranging':
            return 'aida', 'Movement_TransportArtifact'
    return origin_onto, origin_type


def add_rich_arguments(csr, csr_evm, rich_evm, rich_entities, provided_tokens):
    arguments = rich_evm['arguments']

    for argument in arguments:
        entity_id = argument['entityId']
        roles = argument['roles']
        arg_ent = rich_entities[entity_id]

        if provided_tokens:
            arg_span, arg_text = recover_via_token(
                provided_tokens, arg_ent['tokens'])
            arg_head_span, _ = recover_via_token(
                provided_tokens, arg_ent['headWord']['tokens'])
        else:
            arg_span = arg_ent['span']
            arg_head_span = arg_ent['headWord']['span']
            arg_text = arg_ent['text']

        for role in roles:
            onto_name, role_name = role.split(':')

            arg_onto = None
            component = None
            if onto_name == 'fn':
                arg_onto = "framenet"
                frame_name = rich_evm['frame']
                component = 'Semafor'
                # role_pair = (frame_name, role_name)
                full_role_name = frame_name + '_' + role_name
            elif onto_name == 'pb':
                arg_onto = "propbank"
                component = 'Fanse'
                # role_pair = ('pb', role_name)
                full_role_name = 'pb_' + role_name

            if arg_onto and component:
                csr.add_event_arg_by_span(
                    csr_evm, arg_head_span, arg_span, arg_text,
                    arg_onto, full_role_name, component=component
                )


def add_rich_events(rich_event_file, csr, provided_tokens=None):
    base_component_name = 'opera.events.mention.tac.hector'

    with open(rich_event_file) as fin:
        rich_event_info = json.load(fin)

        rich_entities = {}
        csr_entities = {}
        same_head_entities = {}

        for rich_ent in rich_event_info['entityMentions']:
            eid = rich_ent['id']
            rich_entities[eid] = rich_ent

            if provided_tokens:
                span, text = recover_via_token(
                    provided_tokens, rich_ent['tokens'])
                head_span, head_text = recover_via_token(
                    provided_tokens, rich_ent['headWord']['tokens'])
            else:
                span = rich_ent['span']
                text = rich_ent['text']
                head_span = rich_ent['headWord']['span']

            ent = csr.get_by_span(csr.entity_key, span)
            head_ent = csr.get_by_span(csr.entity_head_key, head_span)

            if head_ent:
                same_head_entities[eid] = head_ent

            if not ent:
                if rich_ent.get('component') == 'FrameBasedEventDetector':
                    component = 'Semafor'
                else:
                    component = base_component_name

                ent = csr.add_entity_mention(
                    head_span, span, text, 'conll', rich_ent.get('type', None),
                    entity_form=rich_ent.get('entity_form', 'named'),
                    component=component)

            if 'negationWord' in rich_ent:
                ent.add_modifier('NEG', rich_ent['negationWord'])

            if ent:
                csr_entities[eid] = ent
            else:
                if len(text) > 20:
                    logging.warning("Argument mention {} rejected.".format(eid))
                else:
                    logging.warning(
                        ("Argument mention {}:{} rejected.".format(eid, text)))

        evm_by_id = {}
        for rich_evm in rich_event_info['eventMentions']:
            if provided_tokens:
                span, text = recover_via_token(provided_tokens,
                                               rich_evm['tokens'])
                head_span, head_text = recover_via_token(
                    provided_tokens, rich_evm['headWord']['tokens'])
            else:
                span = rich_evm['span']
                text = rich_evm['text']
                head_span = rich_evm['headWord']['span']

            ontology = 'tac'

            if rich_evm['component'] == "FrameBasedEventDetector":
                component = 'Semafor'
                ontology = 'framenet'
            else:
                component = base_component_name

            ontology, evm_type = fix_event_type_from_frame(
                ontology, rich_evm['type'], rich_evm.get('frame', ''))

            arguments = rich_evm['arguments']

            # Check which entities are in the arguments.
            arg_entity_types = set()
            for argument in arguments:
                entity_id = argument['entityId']

                if entity_id not in csr_entities:
                    logging.error(
                        "Argument entity id {} not found in entity set, "
                        "at doc: [{}]".format(entity_id, rich_event_file))
                    continue

                csr_ent = csr_entities[entity_id]

                for t in csr_ent.get_types():
                    arg_entity_types.add(t)

                if entity_id in same_head_entities:
                    same_head_ent = same_head_entities[entity_id]
                    for t in same_head_ent.get_types():
                        arg_entity_types.add(t)

            csr_evm = csr.add_event_mention(
                head_span, span, text, ontology, evm_type,
                realis=rich_evm.get('realis', None),
                component=component, arg_entity_types=arg_entity_types
            )

            if csr_evm:
                if 'negationWord' in rich_evm:
                    csr_evm.add_modifier('NEG', rich_evm['negationWord'])

                if 'modalWord' in rich_evm:
                    csr_evm.add_modifier('MOD', rich_evm['modalWord'])

                eid = rich_evm['id']
                evm_by_id[eid] = csr_evm

                add_rich_arguments(
                    csr, csr_evm, rich_evm, rich_entities, provided_tokens
                )

        for relation in rich_event_info['relations']:
            if relation['relationType'] == 'event_coreference':
                args = [('member', evm_by_id[i].id) for i in
                        relation['arguments'] if i in evm_by_id]

                csr.add_relation(
                    'aida', args, 'event_coreference', base_component_name
                )

            if relation['relationType'] == 'entity_coreference':
                args = [('member', csr_entities[i].id) for i in
                        relation['arguments'] if i in csr_entities]

                csr.add_relation('aida', args, 'entity_coreference', 'corenlp')


def load_salience(salience_folder):
    raw_data_path = os.path.join(salience_folder, 'data.json')
    salience_event_path = os.path.join(salience_folder, 'output_event.json')
    salience_entity_path = os.path.join(salience_folder, 'output_entity.json')

    events = defaultdict(list)
    entities = defaultdict(list)

    content_field = 'bodyText'

    with open(raw_data_path) as raw_data:
        for line in raw_data:
            raw_data = json.loads(line)
            docno = raw_data['docno']
            for spot in raw_data['spot'][content_field]:
                entities[docno].append({
                    'mid': spot['id'],
                    'wiki': spot['wiki_name'],
                    'span': tuple(spot['span']),
                    'head_span': tuple(spot['head_span']),
                    'score': spot['score'],
                    'text': spot['surface']
                })

            for spot in raw_data['event'][content_field]:
                events[docno].append(
                    {
                        'text': spot['surface'],
                        'span': tuple(spot['span']),
                        'head_span': tuple(spot['head_span']),
                    }
                )

    scored_events = {}
    with open(salience_event_path) as event_output:
        for line in event_output:
            output = json.loads(line)
            docno = output['docno']

            scored_events[docno] = {}

            if docno in events:
                event_list = events[docno]
                for (hid, score), event_info in zip(
                        output[content_field]['predict'], event_list):
                    scored_events[docno][event_info['head_span']] = {
                        'span': event_info['span'],
                        'salience': score,
                        'text': event_info['text']
                    }

    scored_entities = {}
    with open(salience_entity_path) as entity_output:
        for line in entity_output:
            output = json.loads(line)
            docno = output['docno']

            scored_entities[docno] = {}

            if docno in events:
                ent_list = entities[docno]
                for (hid, score), (entity_info) in zip(
                        output[content_field]['predict'], ent_list):

                    link_score = entity_info['score']

                    if link_score > 0.15:
                        scored_entities[docno][entity_info['head_span']] = {
                            'span': entity_info['span'],
                            'mid': entity_info['mid'],
                            'wiki': entity_info['wiki'],
                            'salience': score,
                            'link_score': entity_info['score'],
                            'text': entity_info['text'],
                        }

    return scored_entities, scored_events


def analyze_sentence(text):
    words = nltk.word_tokenize(text)

    negations = []
    for w in words:
        if w in resources.negative_words:
            negations.append(w)

    return negations


def read_source(source_folder, language, aida_ontology, onto_mapper):
    for source_text_path in glob.glob(source_folder + '/*.txt'):
        # Use the empty newline to handle different newline format.
        with open(source_text_path, newline='') as text_in:
            docid = os.path.basename(source_text_path).split('.')[0]
            csr = CSR('Frames_hector_combined', 1, 'data',
                      aida_ontology=aida_ontology, onto_mapper=onto_mapper)

            csr.add_doc(docid, 'text', language)
            text = text_in.read()
            sent_index = 0

            sent_path = source_text_path.replace('.txt', '.sent')
            if os.path.exists(sent_path):
                with open(sent_path) as sent_in:
                    for span_str in sent_in:
                        span_parts = span_str.strip().split(' ')

                        sb, se = span_parts[:2]

                        # sb, se, seg_id, kf = span_str.split(' ')
                        span = (int(sb), int(se))
                        sent_text = text[span[0]: span[1]]

                        if len(span_parts) >= 4:
                            kf = span_parts[3]
                        else:
                            kf = None

                        if kf == '-':
                            kf = None

                        sent = csr.add_sentence(
                            span, text=sent_text, keyframe=kf)

                        negations = analyze_sentence(sent_text)
                        for neg in negations:
                            sent.add_modifier('NEG', neg)

                        sent_index += 1
            else:
                begin = 0
                sent_index = 0

                sent_texts = text.split('\n')
                sent_lengths = [len(t) for t in sent_texts]
                for sent_text, l in zip(sent_texts, sent_lengths):
                    if sent_text.strip():
                        sent = csr.add_sentence((begin, begin + l),
                                                text=sent_text)
                        negations = analyze_sentence(sent_text)
                        for neg in negations:
                            sent.add_modifier('NEG', neg)

                    begin += (l + 1)
                    sent_index += 1

            yield csr, docid


def mid_rdf_format(mid):
    return mid.strip('/').replace('/', '.')


def add_entity_salience(csr, entity_salience_info):
    for span, data in entity_salience_info.items():
        entity = csr.get_by_span(csr.entity_key, span)
        if not entity:
            # Names that can only spot by DBpedia is considered to be nominal.
            entity = csr.add_entity_mention(
                span, data['span'], data['text'], 'aida', None,
                entity_form='nominal', component='dbpedia-spotlight-0.7'
            )

        if not entity:
            if len(data['text']) > 20:
                logging.info("Wikified mention [{}] rejected.".format(span))
            else:
                logging.info(
                    "Wikified mention [{}:{}] rejected.".format(
                        span, data['text'])
                )

        if entity:
            entity.add_salience(data['salience'])
            entity.add_linking(
                mid_rdf_format(data['mid']), data['wiki'], data['link_score'],
                # component='wikifier'
            )


def add_event_salience(csr, event_salience_info):
    for span, data in event_salience_info.items():
        event = csr.get_by_span(csr.event_key, span)
        if not event:
            event = csr.add_event_mention(
                span, data['span'], data['text'],
                'aida', None, component='opera.events.mention.tac.hector')
            if event:
                event.add_salience(data['salience'])


def token_to_span(conll_file):
    tokens = []
    with open(conll_file) as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            elif line == "":
                continue
            else:
                parts = line.split('\t')
                span = tuple([int(s) for s in parts[-1].split(',')])
                tokens.append((span, parts[1]))
    return tokens


def main(config):
    assert config.conllu_folder is not None
    assert config.csr_output is not None

    if not os.path.exists(config.csr_output):
        os.makedirs(config.csr_output)

    aida_ontology = OntologyLoader(config.ontology_path)
    onto_mapper = MappingLoader()
    onto_mapper.load_arg_aida_mapping(config.seedling_argument_mapping)
    onto_mapper.load_event_aida_mapping(config.seedling_event_mapping)

    if config.add_rule_detector:
        # Rule detector should not need existing vocabulary.
        token_vocab = Vocab(config.resource_folder, 'tokens',
                            embedding_path=config.word_embedding,
                            emb_dim=config.word_embedding_dim,
                            ignore_existing=True)
        tag_vocab = Vocab(config.resource_folder, 'tag',
                          embedding_path=config.tag_list,
                          ignore_existing=True)
        detector = DetectionRunner(config, token_vocab, tag_vocab,
                                   aida_ontology)

    ignore_edl = False
    if config.edl_json:
        if os.path.exists(config.edl_json):
            logging.info("Loading from EDL: {}".format(config.edl_json))
        else:
            logging.warning("EDL output not found: {}, will be ignored.".format(
                config.edl_json))
            ignore_edl = True

    for csr, docid in read_source(config.source_folder, config.language,
                                  aida_ontology, onto_mapper):
        logging.info('Working with docid: {}'.format(docid))

        if config.edl_json and not ignore_edl:
            if not os.path.exists(config.edl_json):
                logging.info("No EDL output")
            else:
                edl_file = find_by_id(config.edl_json, docid)
                if edl_file:
                    logging.info("Adding EDL results: {}".format(edl_file))
                    edl_entities = add_edl_entities(edl_file, csr)

                    if config.relation_json:
                        relation_file = find_by_id(config.relation_json, docid)
                        if relation_file:
                            logging.info("Adding relations between entities.")
                            add_entity_relations(
                                relation_file, edl_entities, csr)
                        else:
                            logging.error("Cannot find the relation file for"
                                          " {}".format(docid))

        conll_file = find_by_id(config.conllu_folder, docid)
        if not conll_file:
            logging.warning("CoNLL file for doc {} is missing, please "
                            "check your paths.".format(docid))
            continue

        tokens = None

        if config.rich_event_token:
            tokens = token_to_span(conll_file)

        if config.rich_event:
            if os.path.exists(config.rich_event):
                rich_event_file = find_by_id(config.rich_event, docid)
                if rich_event_file:
                    logging.info(
                        "Adding events with rich output: {}".format(
                            rich_event_file))
                    add_rich_events(rich_event_file, csr, tokens)
            else:
                logging.info("No rich event output.")

        if config.salience_data:
            if not os.path.exists(config.salience_data):
                logging.info("No salience added.")
            else:
                scored_entities, scored_events = load_salience(
                    config.salience_data)
                if docid in scored_entities:
                    add_entity_salience(csr, scored_entities[docid])
                if docid in scored_events:
                    add_event_salience(csr, scored_events[docid])

        logging.info("Reading on CoNLLU: {}".format(conll_file))
        # The conll files may contain information from another language.
        test_reader = ConllUReader([conll_file], config, token_vocab,
                                   tag_vocab, config.language)

        if config.add_rule_detector:
            logging.info("Adding from ontology based rule detector.")
            # Adding rule detector. This is the last detector that use other
            # information from the CSR, including entity and events.
            detector.predict(test_reader, csr, 'maria_multilingual')

        csr.write(os.path.join(config.csr_output, docid + '.csr.json'))


if __name__ == '__main__':
    from event import util
    import sys


    class CombineParams(DetectionParams):
        source_folder = Unicode(help='source text folder').tag(config=True)
        rich_event = Unicode(help='Rich event output.').tag(config=True)
        edl_json = Unicode(help='EDL json output.').tag(config=True)
        relation_json = Unicode(help='Relation json output.').tag(config=True)
        salience_data = Unicode(help='Salience output.').tag(config=True)
        rich_event_token = Bool(
            help='Whether to use tokens from rich event output',
            default_value=False).tag(config=True)
        add_rule_detector = Bool(help='Whether to add rule detector',
                                 default_value=False).tag(config=True)
        output_folder = Unicode(help='Parent output directory').tag(config=True)
        conllu_folder = Unicode(help='CoNLLU directory').tag(config=True)
        csr_output = Unicode(help='Main CSR output directory').tag(config=True)

        # Aida specific
        ontology_path = Unicode(help='Ontology url or path.').tag(config=True)
        seedling_event_mapping = Unicode(
            help='Seedling event mapping to TAC-KBP').tag(config=True)
        seedling_argument_mapping = Unicode(
            help='Seedling argument mapping to TAC-KBP').tag(config=True)


    conf = PyFileConfigLoader(sys.argv[1]).load_config()

    cl_conf = util.load_command_line_config(sys.argv[2:])
    conf.merge(cl_conf)

    params = CombineParams(config=conf)

    log_file = os.path.join(params.output_folder, 'combiner.log')
    print("Logs will be output at {}".format(log_file))
    # util.set_file_log(log_file)

    util.set_basic_log()

    main(params)

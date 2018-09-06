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
import time


def add_entity_relations(relation_file, edl_entities, csr):
    rel_comp = 'opera.relations.xiang'

    with open(relation_file) as rf:
        data = json.load(rf)

        for relation in data:
            for rel in relation['rels']:
                args = []
                arg_names = []
                mention_span = rel['span']
                score = rel['score']
                for arg_name, relen in rel.items():
                    if arg_name == 'rel' or arg_name == 'span' or arg_name == 'score':
                        # Not an argument field.
                        continue

                    if relen not in edl_entities:
                        logging.error(
                            "Relation entities [{}] not found at {}".format(
                                relen, relation_file)
                        )
                        continue

                    args.append(edl_entities[relen].id)
                    arg_names.append(arg_name)

                if len(args) < 2:
                    logging.error(
                        "Insufficient number of fields {} in relation "
                        "at {}. some might be filtered".format(
                            len(args), relation_file))
                    continue

                # Adding a score here will create a interp level score.
                csr_rel = csr.add_relation(
                    arguments=args, arg_names=arg_names,
                    component=rel_comp, span=mention_span,
                    score=score
                )

                if csr_rel:
                    csr_rel.add_type('aida', rel['rel'])
                    # Adding a score here will create facet.
                    # csr_rel.add_type('aida', rel['rel'],score=score)


def add_edl_entities(edl_file, csr):
    edl_component_id = 'opera.entities.edl'

    edl_entities = {}

    if not edl_file:
        return
    with open(edl_file) as edl:
        data = json.load(edl)

        for entity_sent in data:
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


def ok_entity(ent):
    head_pos = ent.get('headWord', None).get('pos', None)

    if not head_pos:
        return False

    if head_pos.startswith('N') or head_pos == 'CD' or head_pos.startswith(
            'PR') or head_pos.startswith('J'):
        return True

    if head_pos == 'IN':
        ent_len = len(ent['text'])
        head_len = len(ent['headWord']['lemma'])
        if ent_len > head_len:
            return True
        else:
            # If the entity only contains IN, it is not OK.
            return False

    return False


def add_rich_arguments(csr, csr_evm, rich_evm, rich_entities, provided_tokens):
    rich_args = rich_evm['arguments']

    for rich_arg in rich_args:
        entity_id = rich_arg['entityId']
        roles = rich_arg['roles']
        rich_arg_ent = rich_entities[entity_id]

        if provided_tokens:
            arg_span, arg_text = recover_via_token(
                provided_tokens, rich_arg_ent['tokens'])
            arg_head_span, _ = recover_via_token(
                provided_tokens, rich_arg_ent['headWord']['tokens'])
        else:
            arg_span = rich_arg_ent['span']
            arg_head_span = rich_arg_ent['headWord']['span']
            arg_text = rich_arg_ent['text']

        arg_comp = rich_arg.get('component', None)

        component = 'opera.events.mention.tac.hector'
        if arg_comp:
            if arg_comp == 'FanseAnnotator':
                component = 'Fanse'
                # Fanse is noisy.
                continue
            elif arg_comp == 'SemaforAnnotator':
                component = 'Semafor'
            elif arg_comp == 'allennlp':
                component = 'AllenNLP.srl'

        for role in roles:
            onto_name, role_name = role.split(':')

            if onto_name == 'fn':
                arg_onto = "framenet"
                frame_name = rich_evm['frame']
                # role_pair = (frame_name, role_name)
                full_role_name = frame_name + '_' + role_name
            elif onto_name == 'pb':
                if component == 'Semafor':
                    # Do not use the propbank role by Semafor.
                    continue

                arg_onto = "propbank"
                if role_name.startswith('R-'):
                    # Ignoring R- style Reference Arguments.
                    # These are normally 'which', 'what', not very useful.
                    continue
                elif role_name.startswith('C-'):
                    # Ignoring C- style Discontinuous Arguments.
                    # These are useful, but hard to represent.
                    continue
                full_role_name = 'pb_' + role_name
            else:
                # Unknown argument ontology
                continue

            if arg_onto and component:
                csr_arg = csr.add_event_arg_by_span(
                    csr_evm, arg_head_span, arg_span, arg_text,
                    arg_onto, full_role_name, component=component
                )

                csr_arg_ent = csr_arg.entity_mention

                if 'entityForm' in rich_arg_ent:
                    csr_arg_ent.set_form(rich_arg_ent['entityForm'])

                if 'negationWord' in rich_arg_ent:
                    csr_arg_ent.add_modifier(
                        'NEG', rich_arg_ent['negationWord'])

                if not ok_entity(rich_arg_ent):
                    # print("not ok ", head_pos, csr_arg_ent.text)
                    csr_arg_ent.set_not_ok()


def add_rich_events(csr, rich_event_file, provided_tokens=None):
    base_component_name = 'opera.events.mention.tac.hector'

    with open(rich_event_file) as fin:
        rich_event_info = json.load(fin)

        rich_entities = {}
        csr_entities = {}

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

            rich_comp = rich_ent.get('component')

            if rich_comp == 'FrameBasedEventDetector':
                component = 'Semafor'
            elif rich_comp == 'StanfordCoreNlpAnnotator':
                component = 'corenlp'
            elif rich_comp == 'allennlp':
                component = base_component_name
            else:
                component = base_component_name

            if 'type' in rich_ent:
                # These are typed entity mentions detected by NER.
                ent = csr.add_entity_mention(
                    head_span, span, text, 'conll', rich_ent['type'],
                    entity_form=rich_ent.get('entityForm', None),
                    component=component
                )

                if 'negationWord' in rich_ent:
                    ent.add_modifier('NEG', rich_ent['negationWord'])

                if ent:
                    csr_entities[eid] = ent
                else:
                    if len(text) > 20:
                        logging.warning(
                            "Argument mention {} rejected.".format(eid))
                    else:
                        logging.warning(
                            ("Argument mention {}:{} "
                             "rejected.".format(eid, text)))

        csr_events = {}
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

            # Collect the entity types of this event's argument.
            arg_entity_types = set()
            for argument in arguments:
                entity_id = argument['entityId']
                if entity_id in csr_entities:
                    csr_ent = csr_entities[entity_id]
                    for t in csr_ent.get_types():
                        arg_entity_types.add(t)

            # Add an event, use the argument entity types to help debug.
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
                csr_events[eid] = csr_evm

                add_rich_arguments(
                    csr, csr_evm, rich_evm, rich_entities, provided_tokens
                )

        for relation in rich_event_info['relations']:
            if relation['relationType'] == 'event_coreference':
                args = [csr_events[i].id for i in relation['arguments'] if
                        i in csr_events]
                csr_rel = csr.add_relation(args, component=base_component_name)
                if csr_rel:
                    csr_rel.add_type('aida', 'event_coreference')

            if relation['relationType'] == 'entity_coreference':
                args = [csr_entities[i].id for i in relation['arguments'] if
                        i in csr_entities]
                csr_rel = csr.add_relation(args, component='corenlp')
                if csr_rel:
                    csr_rel.add_type('aida', 'entity_coreference')


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

            runid = time.strftime("%Y%m%d%H%M")

            csr = CSR('Frames_hector_combined', runid, 'data',
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


def add_entity_linking(csr, wiki_file, lang):
    with open(wiki_file) as f:
        annos = json.load(f).get('annotations', [])
        for anno in annos:
            span = (anno['start'], anno['end'])
            text = anno['spot']
            # It is indeed dbpedia-spotlight-0.7.1.
            entity = csr.add_entity_mention(
                span, span, text, 'aida', None, None,
                component='dbpedia-spotlight-0.7'
            )

            if not entity:
                if len(text) > 20:
                    logging.info("Wiki mention [{}] rejected.".format(span))
                else:
                    logging.info(
                        "Wiki mention [{}:{}] rejected.".format(span, text)
                    )

            if entity:
                if 'mid' in anno:
                    mid = mid_rdf_format(anno['mid'])
                else:
                    mid = None

                entity.add_linking(
                    mid, anno['title'], anno['link_probability'],
                    component='dbpedia-spotlight-0.7', lang=lang
                )


def add_entity_salience(csr, entity_salience_info):
    for span, data in entity_salience_info.items():
        # entity = csr.get_by_span(csr.entity_key, span)
        # if not entity:

        # Names that can only spot by DBpedia is considered to be nominal.
        entity = csr.add_entity_mention(
            span, data['span'], data['text'], 'aida', None,
            None, component='dbpedia-spotlight-0.7'
        )

        if not entity:
            if len(data['text']) > 20:
                logging.info("Salience mention [{}] rejected.".format(span))
            else:
                logging.info(
                    "Salience mention [{}:{}] rejected.".format(
                        span, data['text'])
                )

        if entity:
            entity.add_salience(data['salience'])
            entity.add_linking(
                mid_rdf_format(data['mid']), data['wiki'], data['link_score'],
                component='dbpedia-spotlight-0.7'
            )


def add_event_salience(csr, event_salience_info):
    for span, data in event_salience_info.items():
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


def read_parent_child_info(root_file):
    first_line = True

    child2root = {}

    with open(root_file) as infile:
        for line in infile:
            if first_line:
                first_line = False
                continue

            fields = line.strip().split('\t')
            parent_uid = fields[0]
            child_file = fields[1]

            child2root[child_file.split('.')[0]] = parent_uid

    return child2root


def main(config):
    assert config.conllu_folder is not None
    assert config.csr_output is not None

    if not os.path.exists(config.csr_output):
        os.makedirs(config.csr_output)

    aida_ontology = OntologyLoader(config.ontology_path)
    onto_mapper = MappingLoader()
    onto_mapper.load_arg_aida_mapping(config.seedling_argument_mapping)
    onto_mapper.load_event_aida_mapping(config.seedling_event_mapping)

    if config.parent_children_tab:
        child2root = read_parent_child_info(config.parent_children_tab)
    else:
        child2root = {}

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

        root_id = child2root.get(docid, None)
        csr.set_root(root_id)

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
                                relation_file, edl_entities, csr
                            )
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
                    add_rich_events(csr, rich_event_file, tokens)
            else:
                logging.info("No rich event output.")

        if config.dbpedia_wiki_json:
            if os.path.exists(config.dbpedia_wiki_json):
                wiki_file = find_by_id(config.dbpedia_wiki_json, docid)
                if wiki_file:
                    logging.info(
                        "Adding wiki linking from dbpedia spotlight: {}".format(
                            wiki_file))
                    add_entity_linking(csr, wiki_file, config.language)

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
        dbpedia_wiki_json = Unicode(help='DBpedia output').tag(config=True)
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
        parent_children_tab = Unicode(
            help='File parent children relations provided').tag(config=True)


    conf = PyFileConfigLoader(sys.argv[1]).load_config()

    cl_conf = util.load_command_line_config(sys.argv[2:])
    conf.merge(cl_conf)

    params = CombineParams(config=conf)

    log_file = os.path.join(params.output_folder, 'combiner.log')
    print("Logs will be output at {}".format(log_file))
    util.set_file_log(log_file)

    # util.set_basic_log()

    # TODO: add root

    main(params)

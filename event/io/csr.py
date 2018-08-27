import json
import os
from collections import defaultdict, Counter
import logging
import string
from event.util import remove_punctuation


class Constants:
    GENERAL_ENTITY_TYPE = 'aida:General_Entity'
    GENERAL_EVENT_TYPE = 'aida:General_Event'
    GENERAL_REL_TYPE = 'aida:General_Rel'


entity_type_mapping = {
    "Fac": "Facility",
    "Gpe": "GeopoliticalEntity",
    "Loc": "Location",
    # "Nom": "Nominal",
    "Org": "Organization",
    "Per": "Person",
    "Veh": "Vehicle",
    "Wea": "Weapon",
}


def fix_entity_type(t):
    t_l = t.lower().title()
    if t_l in entity_type_mapping:
        return entity_type_mapping[t_l]
    else:
        return t


class Jsonable:
    def json_rep(self):
        raise NotImplementedError

    def __str__(self):
        return json.dumps(self.json_rep())


class Frame(Jsonable):
    """
    Represent top level frame collection.
    """

    def __init__(self, fid, frame_type, parent, component=None, score=None):
        self.type = frame_type
        self.id = fid
        self.parent = parent
        self.component = component
        self.score = score

    def json_rep(self):
        rep = {
            '@type': self.type,
        }

        if self.id:
            rep['@id'] = self.id

        # if self.parent:
        #     rep['parent_scope'] = self.parent

        if self.component:
            rep['component'] = self.component

        if self.score:
            rep['score'] = self.score

        return rep


class Document(Frame):
    """
    Represent a document frame.
    """

    def __init__(self, fid, doc_name, media_type, language):
        super().__init__(fid, 'document', None)
        self.doc_name = doc_name
        self.media_type = media_type
        self.language = language
        self.num_sentences = 0

    def json_rep(self):
        rep = super().json_rep()
        rep['media_type'] = self.media_type
        rep['language'] = self.language
        rep['num_sentences'] = self.num_sentences
        return rep


class Interp(Jsonable):
    """
    An interp is an interpretation of the evidence. It can contain values like
    event/entity type, database links, etc.
    """

    def __init__(self, interp_type):
        self.interp_type = interp_type
        self.__fields = defaultdict(lambda: defaultdict(dict))
        self.multi_value_fields = set()

    def get_field(self, name):
        if name in self.__fields:
            return dict(self.__fields.get(name))
        else:
            return None

    def modify_field(self, name, key_name, content_rep, content, component=None,
                     score=None):
        original = self.__fields[name][key_name][content_rep]

        if not component:
            component = original['component']

        if not score:
            score = original['score']

        self.__fields[name][key_name][content_rep] = {
            'content': content, 'score': score, 'component': component,
        }

    def add_fields(self, name, key_name, content_rep, content,
                   component=None, score=None, multi_value=False):
        """
        Add an interp field
        :param name: The name to be displayed for this field.
        :param key_name: A unique key name for this field.
        :param content_rep: A value to identify this content from the rest,
        if two content_reps are the same, the are consider the same field.
        :param content: The actual content.
        :param component: The component name.
        :param score: The score of this interp field.
        :param multi_value: Whether this field can contain multiple values.
        :return:
        """
        # If the key and the corresponding content are the same,
        # then the two interps are consider the same.
        self.__fields[name][key_name][content_rep] = {
            'content': content, 'component': component, 'score': score
        }
        if multi_value:
            self.multi_value_fields.add(name)

    def is_empty(self):
        return len(self.__fields) == 0

    def json_rep(self):
        rep = {
            '@type': self.interp_type,
        }
        for field_name, field_info in self.__fields.items():
            field_data = []

            for key_name, keyed_content in field_info.items():
                if len(keyed_content) > 1:
                    # Multiple interpretation found.
                    field = {'@type': 'xor', 'args': []}

                    for key, value in keyed_content.items():
                        v = value['content']
                        score = value['score']
                        component = value['component']
                        v_str = v.json_rep() if \
                            isinstance(v, Jsonable) else str(v)
                        r = {
                            '@type': 'facet',
                            'value': v_str,
                        }
                        if score:
                            r['score'] = score
                        if component:
                            r['component'] = component

                        field['args'].append(r)
                else:
                    # Single interpretation.
                    for key, value in keyed_content.items():
                        v = value['content']
                        score = value['score']
                        component = value['component']
                        v_str = v.json_rep() if \
                            isinstance(v, Jsonable) else str(v)

                        # Facet repr is too verbose.
                        if score:
                            field = {'@type': 'facet', 'value': v_str}
                            if score:
                                field['score'] = score
                            if component:
                                field['component'] = component
                        else:
                            field = v_str

                field_data.append(field)

            if field_name not in self.multi_value_fields:
                field_data = field_data[0]
            rep[field_name] = field_data
        return rep


class InterpFrame(Frame):
    """
    A frame that can contain interpretations.
    """

    def __init__(self, fid, frame_type, parent, interp_type, component=None):
        super().__init__(fid, frame_type, parent, component)
        self.interp_type = interp_type
        self.interp = Interp(interp_type)

    def json_rep(self):
        rep = super().json_rep()
        if self.interp and not self.interp.is_empty():
            rep['interp'] = self.interp.json_rep()
            # for interp in self.interps:
            #     rep['interp'].update(interp.json_rep())
        return rep


class RelArgFrame(Frame):
    """
    A frame for the arguments of a relation.
    """

    def __init__(self, fid):
        super().__init__(fid, 'argument', None)
        self.members = []

    def add_arg(self, arg):
        self.members.append(arg)

    def json_rep(self):
        rep = []
        for arg_type, arg_ent in self.members:
            rep.append(
                {
                    '@type': 'argument',
                    'type': 'aida:' + arg_type,
                    'arg': arg_ent
                }
            )
        return rep


class ValueFrame(Frame):
    """
    A frame that mainly contain values.
    """

    def __init__(self, fid, frame_type, component=None, score=None):
        super().__init__(fid, frame_type, None, component=component,
                         score=score)


class Span:
    """
    Represent text span (begin and end), according to the reference.
    """

    def __init__(self, reference, begin, length):
        self.reference = reference
        self.begin = begin
        self.length = length

    def json_rep(self):
        return {
            '@type': 'text_span',
            'reference': self.reference,
            'start': self.begin,
            'length': self.length,
        }

    def get(self):
        return self.begin, self.begin + self.length

    def __str__(self):
        return "%s: %d,%d" % (
            self.reference, self.begin, self.begin + self.length
        )


class SpanInterpFrame(InterpFrame):
    """
    Commonly used frame to represent a mention, has both spans and
    interpretations.
    """

    def __init__(self, fid, frame_type, parent, interp_type, reference, begin,
                 length, text, component=None):
        super().__init__(fid, frame_type, parent, interp_type, component)
        self.span = Span(reference, begin, length)
        self.text = text
        self.modifiers = {}

    def add_modifier(self, modifier_type, modifier_text):
        self.modifiers[modifier_type] = modifier_text

    def json_rep(self):
        rep = super().json_rep()
        info = {
            'provenance': {
                '@type': 'text_span',
                'reference': self.span.reference,
                'start': self.span.begin,
                'length': self.span.length,
                'text': self.text,
                'parent_scope': self.parent,
                'modifiers': self.modifiers,
            }
        }
        rep.update(info)
        return rep


class Sentence(SpanInterpFrame):
    """
    Represent a sentence.
    """

    def __init__(self, fid, parent, reference, begin, length,
                 text, component=None, keyframe=None):
        super().__init__(fid, 'sentence', parent, 'sentence_interp', reference,
                         begin, length, text, component=component)
        self.keyframe = keyframe

    def substring(self, span):
        begin = span[0] - self.span.begin
        end = span[1] - self.span.begin
        return self.text[begin: end]

    def json_rep(self):
        rep = super().json_rep()
        if self.keyframe:
            rep['@keyframe'] = self.keyframe
        return rep


class EntityMention(SpanInterpFrame):
    """
    Represent a entity mention (in output, it is called entity_evidence).
    """

    def __init__(self, fid, parent, reference, begin, length, text,
                 component=None):
        super().__init__(fid, 'entity_evidence', parent,
                         'entity_evidence_interp', reference, begin, length,
                         text, component)
        self.entity_types = []

    def add_form(self, entity_form):
        self.interp.add_fields('form', 'form', entity_form, entity_form)

    def get_types(self):
        return self.entity_types

    def add_type(self, ontology, entity_type, score=None, component=None):
        # type_interp = Interp(self.interp_type)
        if entity_type == "null":
            return

        entity_type = fix_entity_type(entity_type)
        onto_type = ontology + ":" + entity_type

        if component == self.component:
            # Inherit frame component name.
            component = None

        self.interp.add_fields('type', 'type', onto_type, onto_type,
                               score=score, component=component)

        self.entity_types.append(onto_type)
        # input("Added entity type for {}, {}".format(self.id, self.text))

    def add_linking(self, mid, wiki, score, lang='en', component=None):
        if mid.startswith('/'):
            mid = mid.strip('/')

        fb_xref = ValueFrame('freebase:' + mid, 'db_reference', score=score)
        self.interp.add_fields('xref', 'freebase', mid, fb_xref,
                               multi_value=True)

        wiki_xref = ValueFrame(lang + '_wiki:' + wiki, 'db_reference',
                               score=score)
        self.interp.add_fields('xref', 'wikipedia', wiki, wiki_xref,
                               component=component, multi_value=True)

    def add_salience(self, salience_score):
        self.interp.add_fields('salience', 'score', 'score', salience_score)


class Argument(Frame):
    """
    An argument of event, which is simply a wrap around an entity,
    """

    def __init__(self, arg_role, entity_mention, fid, component=None):
        super().__init__(fid, 'argument', None, component=component)
        self.entity_mention = entity_mention
        self.arg_role = arg_role

    def json_rep(self):
        rep = super().json_rep()
        rep['type'] = self.arg_role
        rep['text'] = self.entity_mention.text
        rep['arg'] = self.entity_mention.id

        return rep


class EventMention(SpanInterpFrame):
    """
    An event mention (in output, it is called event_evidence)
    """

    def __init__(self, fid, parent, reference, begin, length, text,
                 component=None):
        super().__init__(fid, 'event_evidence', parent, 'event_evidence_interp',
                         reference, begin, length, text, component=component)
        self.trigger = None
        self.event_type = None
        self.realis = None

    def add_trigger(self, begin, length):
        self.trigger = Span(self.span.reference, begin, length)

    def add_interp(self, ontology, event_type, realis, score=None,
                   component=None):
        # interp = Interp(self.interp_type)
        onto_type = ontology + ":" + event_type

        if component == self.component:
            # Inherit frame component name.
            component = None

        self.interp.add_fields('type', 'type', onto_type, onto_type,
                               score=score, component=component)
        self.interp.add_fields('realis', 'realis', realis, realis, score=score,
                               component=component)
        self.event_type = onto_type
        self.realis = realis

    def add_arg(self, ontology, arg_role, entity_mention, arg_id,
                score=None, component=None):
        arg = Argument(ontology + ':' + arg_role, entity_mention, arg_id,
                       component=component)
        # If we use the role name as as the key name, if there are multiple
        # entity mentions being identified for this field we consider them as
        # alternative interpretation.
        self.interp.add_fields(
            'args', arg_role, entity_mention.id, arg, score=score,
            component=component, multi_value=True
        )

    def add_salience(self, salience_score):
        self.interp.add_fields('salience', 'score', 'score', salience_score)


class RelationMention(InterpFrame):
    """
    Represent a relation between frames (more than 1).
    """

    def __init__(self, fid, ontology, relation_type, arguments, score=None,
                 component=None):
        super().__init__(fid, 'relation_evidence', None,
                         'relation_evidence_interp', component=component)
        self.relation_type = relation_type
        self.arguments = arguments
        onto_type = ontology + ":" + relation_type
        self.interp.add_fields('type', 'type', onto_type, onto_type)

    def json_rep(self):
        arg_frame = RelArgFrame(None)
        for arg in self.arguments:
            arg_frame.add_arg(arg)

        self.interp.add_fields('args', 'args', self.relation_type, arg_frame)

        return super().json_rep()

    def add_arg(self, arg):
        self.arguments.append(arg)


class CSR:
    """
    Main class that collect and output frames.
    """

    def __init__(self, component_name, run_id, namespace,
                 media_type='text', aida_ontology=None, onto_mapper=None):
        self.header = {
            "@context": [
                "https://www.isi.edu/isd/LOOM/opera/"
                "jsonld-contexts/resources.jsonld",
                "https://www.isi.edu/isd/LOOM/opera/"
                "jsonld-contexts/ail/0.3/frames.jsonld",
            ],
            '@type': 'frame_collection',
            'meta': {
                '@type': 'meta_info',
                'component': component_name,
                'organization': 'CMU',
                'media_type': media_type,
            },
        }

        self._docs = {}

        self.current_doc = None

        self._span_frame_map = defaultdict(dict)
        self._frame_map = defaultdict(dict)
        self._char_sent_map = {}

        self.run_id = run_id

        self.__index_store = Counter()

        self.ns_prefix = namespace
        self.media_type = media_type

        self.entity_key = 'entity'
        self.entity_head_key = 'entity_head'
        self.entity_group_key = 'entity_group'
        self.event_key = 'event'
        self.event_group_key = 'event_group'
        self.sent_key = 'sentence'
        self.rel_key = 'relation'

        self.base_onto_name = 'aida'

        self.onto_mapper = onto_mapper

        if aida_ontology:
            self.event_onto = aida_ontology.event_onto_text()
            self.arg_restricts = aida_ontology.get_arg_restricts()

            self.canonical_types = {}
            for event_type in self.event_onto:
                c_type = onto_mapper.canonicalize_type(event_type)
                self.canonical_types[c_type] = event_type

            for _, arg_type in self.arg_restricts:
                c_type = onto_mapper.canonicalize_type(arg_type)
                self.canonical_types[c_type] = arg_type

    def load_from_file(self, csr_file):
        def compute_span(this_frame):
            parent_sent_id = this_frame['provenance']['reference']
            if sent_ref in self._frame_map[self.sent_key]:
                parent_sent = self._frame_map[self.sent_key][parent_sent_id]
                offset = parent_sent.span.begin
            else:
                offset = 0
            s = this_frame["provenance"]["start"] + offset
            e = this_frame["provenance"]["length"] + s
            return (s, e)

        with open(csr_file) as fin:
            csr_json = json.load(fin)
            docid = csr_json['meta']['document_id']
            media_type = csr_json['meta']['media_type']
            docname = ''.join(docid.split(':')[1].split('-')[:-1])

            frame_data = defaultdict(list)

            for frame in csr_json['frames']:
                frame_type = frame['@type']
                fid = frame['@id']

                if frame_type == 'document':
                    self.add_doc(docname, media_type, frame['language'])
                elif frame_type == 'sentence':
                    start = frame["provenance"]["start"]
                    end = frame["provenance"]["length"] + start
                    self.add_sentence(
                        (start, end), frame["provenance"]['text'], sent_id=fid)
                else:
                    frame_data[frame_type].append(frame)

            entities = {}
            for frame in frame_data['entity_evidence']:
                sent_ref = frame['provenance']['reference']
                interp = frame['interp']
                onto, t = interp['type'].split(':')
                text = frame["provenance"]['text']
                span = compute_span(frame)
                eid = frame['@id']

                csr_ent = self.add_entity_mention(
                    span, span, text, onto, t, sent_ref, interp['form'],
                    frame['component'], entity_id=eid
                )

                entities[eid] = csr_ent

            for frame in frame_data['event_evidence']:
                sent_ref = frame['provenance']['reference']
                interp = frame['interp']
                onto, t = interp['type'].split(':')
                text = frame["provenance"]['text']
                span = compute_span(frame)

                csr_evm = self.add_event_mention(
                    span, span, text, onto, t, realis=interp['realis'],
                    sent_id=sent_ref, component=frame['component'],
                    event_id=frame['@id']
                )

                for mod, v in frame["provenance"]["modifiers"].items():
                    csr_evm.add_modifier(mod, v)

                for arg in interp.get('args', []):
                    arg_values = []
                    if arg['@type'] == 'xor':
                        for sub_arg in arg['args']:
                            arg_values.append(sub_arg['value'])
                    else:
                        arg_values.append(arg)

                    for arg_val in arg_values:
                        arg_onto, arg_type = arg_val['type'].split(':')
                        arg_id = arg_val['@id']
                        arg_entity_id = arg_val['arg']
                        ent = entities[arg_entity_id]

                        csr_evm.add_arg(
                            arg_onto, arg_type, ent, arg_id,
                            component=arg_val['component']
                        )

            for frame in frame_data['relation_evidence']:
                fid = frame['@id']
                interp = frame['interp']
                onto, rel_type = interp['type'].split(':')

                args = [arg['arg'] for arg in interp['args']]

                self.add_relation(
                    onto, args, rel_type, component=frame['component'],
                    relation_id=fid
                )

    def __canonicalize_event_type(self):
        canonical_map = {}
        for event_type in self.event_onto:
            c_type = self.__cannonicalize_one_type(event_type)
            canonical_map[c_type] = event_type
        return canonical_map

    def __cannonicalize_one_type(self, event_type):
        return remove_punctuation(event_type).lower()

    def clear(self):
        self._span_frame_map.clear()
        self._frame_map.clear()
        self._char_sent_map.clear()

    def add_doc(self, doc_name, media_type, language):
        ns_docid = self.ns_prefix + ":" + doc_name + "-" + media_type

        if ns_docid in self._docs:
            return

        doc = Document(ns_docid, doc_name, media_type, language)
        self.current_doc = doc
        self._docs[ns_docid] = doc

    def get_events_mentions(self):
        return self.get_frames(self.event_key)

    def get_frames(self, key_type):
        # if key_type not in self._frame_map:
        #     raise KeyError('Unknown frame type : {}'.format(key_type))
        return self._frame_map[key_type]

    def get_frame(self, key_type, frame_id):
        frames = self.get_frames(key_type)
        return frames.get(frame_id)

    def add_sentence(self, span, text=None, component=None, keyframe=None,
                     sent_id=None):
        if not sent_id:
            sent_id = self.get_id('sent')

        if sent_id in self._frame_map[self.sent_key]:
            return sent_id

        docid = self.current_doc.id
        self.current_doc.num_sentences += 1
        sent_text = text if text else ""
        sent = Sentence(sent_id, docid, docid, span[0], span[1] - span[0],
                        text=sent_text, component=component, keyframe=keyframe)
        self._frame_map[self.sent_key][sent_id] = sent

        for c in range(span[0], span[1]):
            self._char_sent_map[c] = sent_id

        return sent

    def set_sentence_text(self, sent_id, text):
        self._frame_map[self.sent_key][sent_id].text = text

    def align_to_text(self, span, text, sent_id=None):
        """
        Hacking the correct spans.
        :param span: The origin mention span.
        :param text: The origin mention text.
        :param sent_id: The sentence if, if provided.
        :return:
        """

        span = tuple(span)
        if not sent_id:
            # Find the sentence if not provided.
            res = self.fit_to_sentence(span)
            if res:
                sent_id, fitted_span = res
        else:
            # Use the provided ones.
            sent_id = sent_id
            fitted_span = span

        if not sent_id or not fitted_span:
            # No suitable sentence found.
            logging.warning(
                "No suitable sentence for entity {}".format(span))
        else:
            sent = self._frame_map[self.sent_key][sent_id]
            if not fitted_span == span:
                # If the fitted span changes, we will simply use the doc text.
                valid_text = sent.substring(fitted_span)
            else:
                # The fitted span has not changed, validate it.
                valid_text = self.validate_span(sent_id, fitted_span, text)

            if valid_text is not None:
                return sent_id, fitted_span, valid_text

    def validate_span(self, sent_id, span, text):
        sent = self._frame_map[self.sent_key][sent_id]
        span_text = sent.substring(span)

        if not span_text == text:
            # logging.warning("Was {}".format(span))
            # logging.warning("Becomes {},{}".format(begin, end))
            # logging.warning(sent_text)
            logging.warning(
                "Span text: [{}] not matching given text [{}]"
                ", at span [{}] at sent [{}]".format(
                    span_text, text, span, sent_id)
            )

            if "".join(span_text.split()) == "".join(text.split()):
                logging.warning('only white space difference, accepting.')
                return span_text
            return None

        return text

    def fit_to_sentence(self, span):
        if span[0] in self._char_sent_map:
            sent_id = self._char_sent_map[span[0]]
            sentence = self._frame_map[self.sent_key][sent_id]
            sent_begin = sentence.span.begin
            sent_end = sentence.span.length + sent_begin

            if span[1] <= sent_end:
                return sent_id, span
            else:
                logging.warning("Force span {} end at sentence end {} of "
                                "{}".format(span, sent_end, sent_id))
                return sent_id, (span[0], sent_end)

    def get_by_span(self, object_type, span):
        span = tuple(span)
        if span in self._span_frame_map[object_type]:
            eid = self._span_frame_map[object_type][span]
            e = self._frame_map[object_type][eid]
            return e
        return None

    def map_entity_type(self, onto_name, entity_type):
        if onto_name == 'conll':
            if entity_type == 'DATE':
                return 'aida', 'Time'
        return onto_name, entity_type

    def add_entity_mention(self, head_span, span, text, ontology, entity_type,
                           sent_id=None, entity_form=None, component=None,
                           entity_id=None):
        if span is None:
            logging.warning("None span provided")
            return

        if text is None:
            logging.warning("None text provided")
            return

        if ontology is None:
            logging.warning("None ontology provided")
            return

        head_span = tuple(head_span)
        align_res = self.align_to_text(span, text, sent_id)

        if align_res:
            sent_id, fitted_span, valid_text = align_res

            sentence_start = self._frame_map[self.sent_key][sent_id].span.begin

            if not entity_id:
                entity_id = self.get_id('ent')

            entity_mention = EntityMention(
                entity_id, sent_id, sent_id,
                fitted_span[0] - sentence_start,
                fitted_span[1] - fitted_span[0], valid_text,
                component=component)
            self._span_frame_map[self.entity_key][fitted_span] = entity_id
            self._frame_map[self.entity_key][entity_id] = entity_mention

            self._span_frame_map[self.entity_head_key][head_span] = entity_id
            self._frame_map[self.entity_head_key][entity_id] = entity_mention
        else:
            return

        ontology, entity_type = self.map_entity_type(ontology, entity_type)

        if entity_form:
            entity_mention.add_form(entity_form)
        if entity_type:
            entity_mention.add_type(ontology, entity_type, component=component)
        else:
            entity_mention.add_type('conll', 'MISC', component=component)
        return entity_mention

    def map_event_type(self, evm_type, onto_name):
        if self.onto_mapper:
            if onto_name == 'tac':
                mapped_evm_type = self.__cannonicalize_one_type(evm_type)
                if mapped_evm_type in self.canonical_types:
                    return self.canonical_types[mapped_evm_type]
            elif self.onto_mapper:
                full_type = onto_name + ':' + evm_type
                event_map = self.onto_mapper.get_aida_event_map()
                if full_type in event_map:
                    return self.canonical_types[event_map[full_type]]
        return None

    def add_event_mention(self, head_span, span, text, onto_name, evm_type,
                          realis=None, sent_id=None, component=None,
                          arg_entity_types=None, event_id=None):
        # Annotation on the same span will be reused.
        head_span = tuple(head_span)

        align_res = self.align_to_text(span, text, sent_id)

        if align_res:
            sent_id, fitted_span, valid_text = align_res

            if head_span in self._span_frame_map[self.event_key]:
                # logging.info("Cannot handle overlapped event mentions now.")
                return

            if not event_id:
                event_id = self.get_id('evm')

            self._span_frame_map[self.event_key][head_span] = event_id

            sent = self._frame_map[self.sent_key][sent_id]

            relative_begin = fitted_span[0] - sent.span.begin
            length = fitted_span[1] - fitted_span[0]

            evm = EventMention(event_id, sent_id, sent_id, relative_begin,
                               length, valid_text, component=component)
            evm.add_trigger(relative_begin, length)
            self._frame_map[self.event_key][event_id] = evm
        else:
            return

        if evm_type:
            realis = 'UNK' if not realis else realis

            mapped_type = self.map_event_type(evm_type, onto_name)

            if arg_entity_types:
                mapped_type = fix_event_type_from_entity(
                    mapped_type, arg_entity_types
                )

            if mapped_type:
                evm.add_interp('aida', mapped_type, realis, component=component)
            else:
                evm.add_interp(onto_name, evm_type, realis, component=component)

        return evm

    def get_id(self, prefix, index=None):
        if not index:
            index = self.__index_store[prefix]
            self.__index_store[prefix] += 1

        return '%s:%s-%s-text-cmu-r%s-%d' % (
            self.ns_prefix,
            prefix,
            self.current_doc.doc_name,
            self.run_id,
            index,
        )

    def map_event_arg_type(self, event_onto, evm_type, full_role_name,
                           aida_arg_ent_types):
        map_to_aida = self.onto_mapper.get_aida_arg_map()

        if event_onto == 'aida':
            key = (self.onto_mapper.canonicalize_type(evm_type), full_role_name)

            mapped_arg_type = None

            if key in map_to_aida:
                candidate_aida_types = map_to_aida[key]

                for arg_aid_type, type_res in candidate_aida_types:
                    c_arg_aida_type = self.onto_mapper.canonicalize_type(
                        arg_aid_type)

                    if type_res:
                        if len(aida_arg_ent_types) > 0:
                            match_resitrct = False
                            for t in aida_arg_ent_types:
                                if t in type_res:
                                    match_resitrct = True

                            if not match_resitrct:
                                logging.info(
                                    "arg is rejected because entity type "
                                    "{} cannot fill {}".format(
                                        arg_aid_type, full_role_name))
                                continue
                    if c_arg_aida_type in self.canonical_types:
                        mapped_arg_type = self.canonical_types[c_arg_aida_type]
                        break
                    elif arg_aid_type.startswith('Internal.'):
                        mapped_arg_type = arg_aid_type
                        break
            elif full_role_name == 'pb_ARGM-TMP' or 'Time' in full_role_name:
                arg_aid_type = evm_type + '_Time'
                full_arg = (evm_type, arg_aid_type)
                if full_arg in self.arg_restricts:
                    mapped_arg_type = arg_aid_type
            elif full_role_name == 'pb_ARGM-LOC' or 'Place' in full_role_name:
                arg_aid_type = evm_type + '_Place'
                full_arg = (evm_type, arg_aid_type)
                if full_arg in self.arg_restricts:
                    mapped_arg_type = arg_aid_type

            return mapped_arg_type

    def add_event_arg_by_span(self, evm, arg_head_span, arg_span,
                              arg_text, arg_onto, full_role_name, component,
                              arg_entity_form='named'):
        ent = self.get_by_span(self.entity_key, arg_span)

        if not ent:
            ent = self.get_by_span(self.entity_head_key, arg_head_span)

        if not ent:
            ent = self.add_entity_mention(
                arg_head_span, arg_span, arg_text, 'aida', "argument",
                entity_form=arg_entity_form, component=component
            )

        if ent:
            evm_onto, evm_type = evm.event_type.split(':')
            arg_id = self.get_id('arg')

            aida_arg_entity_types = []

            for t in ent.get_types():
                if t.startswith('aida:'):
                    aida_arg_entity_types.append(t.split(':')[1])

            in_domain_arg = self.map_event_arg_type(
                evm_onto, evm_type, full_role_name, aida_arg_entity_types
            )

            if in_domain_arg:
                if in_domain_arg.startswith('Internal.'):
                    arg_onto, arg_role = in_domain_arg.split('.')
                else:
                    arg_onto = self.base_onto_name
                    arg_role = in_domain_arg
            else:
                arg_role = full_role_name

            evm.add_arg(arg_onto, arg_role, ent, arg_id, component=component)

    def add_event_arg(self, evm, ent, ontology, arg_role, component):
        arg_id = self.get_id('arg')
        evm.add_arg(ontology, arg_role, ent, arg_id, component=component)

    def add_relation(self, ontology, arguments, relation_type, component=None,
                     relation_id=None):
        if not relation_id:
            relation_id = self.get_id('relm')
        rel = RelationMention(relation_id, ontology, relation_type, arguments,
                              component=component)
        self._frame_map[self.rel_key][relation_id] = rel

    def get_json_rep(self):
        rep = {}
        self.header['meta']['document_id'] = self.current_doc.id
        rep.update(self.header)
        rep['frames'] = [self.current_doc.json_rep()]
        for frame_type, frame_info in self._frame_map.items():
            for fid, frame in frame_info.items():
                rep['frames'].append(frame.json_rep())

        return rep

    def write(self, out_path, append=False):
        if append:
            mode = 'a' if os.path.exists(out_path) else 'w'
        else:
            mode = 'w'

        logging.info("Writing data to [%s]" % out_path)

        with open(out_path, mode) as out:
            json.dump(self.get_json_rep(), out, indent=2, ensure_ascii=False)

        self.clear()


def fix_event_type_from_entity(evm_type, arg_entity_types):
    if evm_type == 'Movement.TransportPerson':
        if ('aida:Person' not in arg_entity_types) and (
                'aida:Vehicle' in arg_entity_types):
            evm_type = 'Movement.TransportArtifact'
    return evm_type


if __name__ == '__main__':
    import sys

    # Test loading a csr file
    csr = CSR('Frames_hector_combined', 1, 'data')
    csr.load_from_file(sys.argv[1])
    csr.write(sys.argv[2])

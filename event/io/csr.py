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

    def __init__(self, fid, frame_type, parent, component=None):
        self.type = frame_type
        self.id = fid
        self.parent = parent
        self.component = component

    def json_rep(self):
        rep = {
            '@type': self.type,
        }

        if self.id:
            rep['@id'] = self.id

        if self.component:
            rep['component'] = self.component

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
    event/entity type, database links, scores, etc.
    """

    def __init__(self, interp_type, score=None):
        """

        :param interp_type:
        :param score: It is possible to add a score to interp level.
        """
        self.interp_type = interp_type
        self.__fields = defaultdict(lambda: defaultdict(dict))
        self.multi_value_fields = set()
        self.interp_score = score

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

        # if not score:
        #     score = original['score']

        self.__fields[name][key_name][content_rep] = {
            'content': content, 'score': score, 'component': component,
        }

    def add_field(self, name, key_name, content_rep, content,
                  component=None, score=None, multi_value=False):
        """
        Add an interp field.
        :param name: The name to be displayed for this field.
        :param key_name: A unique key name for this field.
        :param content_rep: A value to identify this content from the rest,
        works like a hash key.
        :param content: The actual content, it could be the same as content_rep
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

        if self.interp_score:
            rep['score'] = self.interp_score

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
                    # We still do this iteration, but it will only return one
                    # single value.
                    for key, value in keyed_content.items():
                        v = value['content']
                        score = value['score']
                        component = value['component']
                        v_str = v.json_rep() if \
                            isinstance(v, Jsonable) else str(v)

                        # Facet repr is too verbose.
                        if score:
                            # Use the verbose facet level stuff.
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

    def __init__(self, fid, frame_type, parent, interp_type, component=None,
                 score=None):
        super().__init__(fid, frame_type, parent, component)
        self.interp_type = interp_type
        self.interp = Interp(interp_type, score)

    def json_rep(self):
        rep = super().json_rep()
        if self.interp and not self.interp.is_empty():
            rep['interp'] = self.interp.json_rep()
        return rep


class RelArgFrame(Frame):
    """
    A frame for the argument of a relation.
    """

    def __init__(self, fid, arg_type, arg_ent_id):
        super().__init__(fid, 'argument', None)
        self.arg_type = arg_type
        self.arg_ent_id = arg_ent_id

    def json_rep(self):
        rep = {
            '@type': 'argument',
            'type': 'aida:' + self.arg_type,
            'arg': self.arg_ent_id
        }
        return rep


class ValueFrame(Frame):
    """
    A frame that mainly contain values.
    """

    def __init__(self, fid, frame_type, component=None, score=None):
        super().__init__(fid, frame_type, None, component=component)
        self.score = score

    def json_rep(self):
        rep = super().json_rep()
        rep['score'] = self.score


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

    def __init__(
            self, fid, frame_type, parent, interp_type, begin,
            length, text, component=None, score=None
    ):
        super().__init__(fid, frame_type, parent, interp_type, component, score)
        if parent:
            self.span = Span(parent.id, begin, length)
        else:
            self.span = None

        self.text = text
        self.modifiers = {}

        if parent and parent.type == 'sentence':
            # Inherit key frame from parents.
            self.keyframe = parent.keyframe
        else:
            self.keyframe = None

    def add_modifier(self, modifier_type, modifier_text):
        self.modifiers[modifier_type] = modifier_text

    def json_rep(self):
        rep = super().json_rep()
        if self.span:
            info = {
                'provenance': {
                    '@type': 'text_span',
                    'reference': self.span.reference,
                    'start': self.span.begin,
                    'length': self.span.length,
                    'text': self.text,
                    'parent_scope': self.parent.id,
                    'modifiers': self.modifiers,
                }
            }

            if self.keyframe:
                info['provenance']['keyframe'] = self.keyframe

            rep.update(info)
        return rep


class Sentence(SpanInterpFrame):
    """
    Represent a sentence.
    """

    def __init__(self, fid, parent, begin, length, text,
                 component=None, keyframe=None):
        super().__init__(fid, 'sentence', parent, 'sentence_interp',
                         begin, length, text, component=component)
        self.keyframe = keyframe

    def substring(self, span):
        begin = span[0] - self.span.begin
        end = span[1] - self.span.begin
        return self.text[begin: end]


class EntityMention(SpanInterpFrame):
    """
    Represent a entity mention (in output, it is called entity_evidence).
    """

    def __init__(self, fid, parent, begin, length, text,
                 component=None, score=None):
        super().__init__(fid, 'entity_evidence', parent,
                         'entity_evidence_interp', begin, length,
                         text, component=component, score=score)
        self.entity_types = []

    def add_form(self, entity_form):
        self.interp.add_field('form', 'form', entity_form, entity_form)

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

        self.interp.add_field('type', 'type', onto_type, onto_type,
                              score=score, component=component)

        self.entity_types.append(onto_type)


    def add_linking(self, mid, wiki, score, lang='en', component=None):
        if mid.startswith('/'):
            mid = mid.strip('/')

        fb_xref = ValueFrame('freebase:' + mid, 'db_reference', score=score)
        self.interp.add_field('xref', 'freebase', mid, fb_xref,
                              multi_value=True)

        wiki_xref = ValueFrame(lang + '_wiki:' + wiki, 'db_reference',
                               score=score)
        self.interp.add_field('xref', 'wikipedia', wiki, wiki_xref,
                              component=component, multi_value=True)

    def add_salience(self, salience_score):
        self.interp.add_field('salience', 'score', 'score', salience_score)


class RelationMention(SpanInterpFrame):
    """
    Represent a relation mention between other mentions.
    """

    def __init__(self, fid, parent, begin, length, text,
                 component=None, score=None):
        super().__init__(
            fid, 'relation_evidence', parent, 'relation_evidence_interp',
            begin, length, text, component=component, score=score)

        self.arguments = []
        self.rel_types = []
        self.score = score

    def get_types(self):
        return self.rel_types

    def add_type(self, ontology, rel_type, score=None, component=None):
        onto_type = ontology + ":" + rel_type
        self.rel_types.append(onto_type)

        if component == self.component:
            # Inherit frame component name.
            component = None

        self.interp.add_field(
            'type', 'type', onto_type, onto_type, component=component,
            score=score
        )

    def add_arg(self, arg_ent, score=None):
        arg_frame = RelArgFrame(None, 'member', arg_ent)
        self.arguments.append(arg_frame)
        self.interp.add_field(
            'args', 'member_%d' % len(self.arguments), arg_ent, arg_frame,
            score=score, multi_value=True
        )

    def add_named_arg(self, arg_name, arg_ent, score=None):
        arg_frame = RelArgFrame(None, arg_name, arg_ent)
        self.arguments.append(arg_frame)
        self.interp.add_field(
            'args', arg_name, arg_ent, arg_frame,
            score=score, multi_value=True
        )


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

    def __init__(self, fid, parent, begin, length, text, component=None,
                 score=None):
        super().__init__(
            fid, 'event_evidence', parent, 'event_evidence_interp',
            begin, length, text, component=component, score=score)
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

        self.interp.add_field('type', 'type', onto_type, onto_type,
                              score=score, component=component)
        self.interp.add_field('realis', 'realis', realis, realis, score=score,
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
        self.interp.add_field(
            'args', arg_role, entity_mention.id, arg, score=score,
            component=component, multi_value=True
        )

    def add_salience(self, salience_score):
        self.interp.add_field('salience', 'score', 'score', salience_score)


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
        self.entity_group_key = 'entity_group'
        self.event_key = 'event'
        self.event_group_key = 'event_group'
        self.sent_key = 'sentence'
        self.rel_key = 'relation'
        self.image_detection_key = 'image_detection'

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
        def get_parent_sent(this_frame):
            parent_sent_id = this_frame['provenance']['reference']
            if parent_sent_id in self._frame_map[self.sent_key]:
                parent_sent = self._frame_map[self.sent_key][parent_sent_id]
                return parent_sent

        def compute_span(this_frame):
            parent_sent = get_parent_sent(this_frame)
            if parent_sent:
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
                parent_sent = get_parent_sent(frame['provenance']['reference'])
                interp = frame['interp']
                onto, t = interp['type'].split(':')
                text = frame["provenance"]['text']
                span = compute_span(frame)
                eid = frame['@id']

                csr_ent = self.add_entity_mention(
                    span, span, text, onto, t, parent_sent, interp['form'],
                    frame['component'], entity_id=eid
                )

                entities[eid] = csr_ent

            for frame in frame_data['event_evidence']:
                parent_sent = get_parent_sent(frame['provenance']['reference'])
                interp = frame['interp']
                onto, t = interp['type'].split(':')
                text = frame["provenance"]['text']
                span = compute_span(frame)

                csr_evm = self.add_event_mention(
                    span, span, text, onto, t, realis=interp['realis'],
                    parent_sent=parent_sent, component=frame['component'],
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
                arg_names = [arg['type'] for arg in interp['args']]

                span = None
                if 'provenance' in frame:
                    start = frame['provenance']['start']
                    end = start + frame['provenance']['length']
                    span = (start, end)

                if len(set(arg_names)) == 1 and arg_names[0] == 'aida:member':
                    # i.e. all arguments are the same, unordered.
                    rel = self.add_relation(
                        args, component=frame['component'],
                        relation_id=fid, span=span,
                        score=frame.get('score', None)
                    )
                else:

                    rel = self.add_relation(
                        args, arg_names, component=frame['component'],
                        relation_id=fid, span=span,
                        score=frame.get('score', None)
                    )

                rel.add_type(onto, rel_type)

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

        self.current_doc.num_sentences += 1
        sent_text = text if text else ""
        sent = Sentence(
            sent_id, self.current_doc, span[0], span[1] - span[0],
            text=sent_text, component=component, keyframe=keyframe
        )
        self._frame_map[self.sent_key][sent_id] = sent

        for c in range(span[0], span[1]):
            self._char_sent_map[c] = sent_id

        return sent

    def set_sentence_text(self, sent_id, text):
        self._frame_map[self.sent_key][sent_id].text = text

    def align_to_text(self, span, text, sent):
        """
        Hacking the correct spans.
        :param span: The origin mention span.
        :param text: The origin mention text.
        :param sent: The sentence if provided.
        :return:
        """

        span = tuple(span)
        if not sent:
            # Find the sentence if not provided.
            res = self.fit_to_sentence(span)
            if res:
                sent, fitted_span = res
        else:
            # Use the provided sentence.
            fitted_span = span

        if not sent or not fitted_span:
            # No suitable sentence found.
            logging.warning(
                "No suitable sentence for entity {}".format(span))
        else:
            if (not fitted_span == span) or (not text):
                # If the fitted span changes, we will simply use the doc text.
                valid_text = sent.substring(fitted_span)
            else:
                # The fitted span has not changed, validate it.
                valid_text = self.validate_span(sent, fitted_span, text)

            if valid_text is not None:
                return sent, fitted_span, valid_text

    def validate_span(self, sent, span, text):
        span_text = sent.substring(span)

        if not span_text == text:
            logging.warning(
                "Span text: [{}] not matching given text [{}]"
                ", at span [{}] at sent [{}]".format(
                    span_text, text, span, sent.id)
            )

            if "".join(span_text.split()) == "".join(text.split()):
                logging.warning('only white space difference, accepting.')
                return span_text
            return None

        return text

    def fit_to_sentence(self, span):
        if span[0] in self._char_sent_map:
            sentence = self._frame_map[self.sent_key][
                self._char_sent_map[span[0]]]
            sent_begin = sentence.span.begin
            sent_end = sentence.span.length + sent_begin

            if span[1] <= sent_end:
                return sentence, span
            else:
                logging.warning("Force span {} end at sentence end {} of "
                                "{}".format(span, sent_end, sentence.id))
                return sentence, (span[0], sent_end)

    def get_by_span(self, object_type, span):
        span = tuple(span)
        head_span_type = object_type + '_head'
        if span in self._span_frame_map[object_type]:
            return self._span_frame_map[object_type][span]
        elif head_span_type in self._span_frame_map:
            if span in self._span_frame_map[head_span_type]:
                return self._span_frame_map[head_span_type][span]

        return None

    def map_entity_type(self, onto_name, entity_type):
        if onto_name == 'conll':
            if entity_type == 'DATE':
                return 'aida', 'Time'
        return onto_name, entity_type

    def add_relation(self, arguments, arg_names=None, component=None,
                     relation_id=None, span=None, score=None):
        """
        Adding a relation mention to CSR. If the span is not provided, it will
        not have a provenance.

        :param ontology: The ontology name of the relation.
        :param relation_type: The relation type for this relation.
        :param arguments: List of arguments (their frame ids).
        :param arg_names: If provided, the arguments will be named
        accordingly.
        :param component: The component name that produces this relation.
        :param relation_id: A unique relation id, the CSR will automatically
            assign one if not provided.
        :param span: A span providing the provenance of the relation mention
        :param score: A score for this relation, normally we don't use score at
            frame level.

        :return: The created relation mention will be returned
        """
        if not relation_id:
            relation_id = self.get_id('relm')

        if span:
            # We didn't provide text for validation, this may result in
            # incorrect spans. But relations are build on other mentions, so
            # it should generally be OK.
            align_res = self.align_to_text(span, None, None)
            if not align_res:
                return

            sent, fitted_span, valid_text = align_res
            sentence_start = self._frame_map[self.sent_key][sent.id].span.begin
            rel = RelationMention(
                relation_id, sent,
                fitted_span[0] - sentence_start,
                fitted_span[1] - fitted_span[0],
                valid_text, component=component,
                score=score
            )
        else:
            # Adding a relation mention without span information.
            rel = RelationMention(
                relation_id, None, 0, 0, '', component=component, score=score)

        if arg_names is None:
            for arg_ent in arguments:
                rel.add_arg(arg_ent)
        else:
            for arg_name, arg_ent in zip(arg_names, arguments):
                rel.add_named_arg(arg_name, arg_ent)

        self._frame_map[self.rel_key][relation_id] = rel

        return rel

    def add_entity_mention(self, head_span, span, text, ontology, entity_type,
                           parent_sent=None, entity_form=None, component=None,
                           entity_id=None, score=None):
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
        align_res = self.align_to_text(span, text, parent_sent)

        if align_res:
            parent_sent, fitted_span, valid_text = align_res
            sentence_start = parent_sent.span.begin

            if fitted_span in self._span_frame_map[self.entity_key]:
                entity_mention = self._span_frame_map[self.entity_key][
                    fitted_span]
            elif head_span in self._span_frame_map[self.entity_key + "_head"]:
                entity_mention = self._span_frame_map[
                    self.entity_key + "_head"][head_span]
            else:
                if not entity_id:
                    entity_id = self.get_id('ent')

                entity_mention = EntityMention(
                    entity_id, parent_sent,
                    fitted_span[0] - sentence_start,
                    fitted_span[1] - fitted_span[0], valid_text,
                    component=component, score=score
                )
                self._frame_map[self.entity_key][entity_id] = entity_mention
                self._span_frame_map[self.entity_key][
                    fitted_span] = entity_mention
                self._span_frame_map[self.entity_key + "_head"][
                    head_span] = entity_mention

                if entity_form:
                    entity_mention.add_form(entity_form)
        else:
            return

        ontology, entity_type = self.map_entity_type(ontology, entity_type)

        if entity_type:
            entity_mention.add_type(ontology, entity_type, component=component)
        else:
            if len(entity_mention.get_types()) < 1:
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
                          realis=None, parent_sent=None, component=None,
                          arg_entity_types=None, event_id=None):
        # Annotation on the same span will be reused.
        head_span = tuple(head_span)
        span = tuple(span)

        align_res = self.align_to_text(span, text, parent_sent)

        if align_res:
            parent_sent, fitted_span, valid_text = align_res

            if span in self._span_frame_map[self.event_key]:
                logging.info("Cannot handle overlapped event mentions now.")
                evm = self._span_frame_map[self.event_key][span]
            elif head_span in self._span_frame_map[self.event_key + '_head']:
                evm = self._span_frame_map[self.event_key + '_head'][head_span]
            else:
                if not event_id:
                    event_id = self.get_id('evm')

                relative_begin = fitted_span[0] - parent_sent.span.begin
                length = fitted_span[1] - fitted_span[0]

                evm = EventMention(
                    event_id, parent_sent, relative_begin, length, valid_text,
                    component=component
                )
                evm.add_trigger(relative_begin, length)

                self._frame_map[self.event_key][event_id] = evm

                self._span_frame_map[self.event_key][span] = evm
                self._span_frame_map[self.event_key + "_head"][head_span] = evm
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

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


conll_to_target = {
    # "TIME": "Time",
    "PERSON": ("ldcOnt", "PER"),
    "LOCATION": ("ldcOnt", "GPE"),
    "ORGANIZATION": {"ldcOnt", "ORG"},
}


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
        # If marked as deleted, the CSR object will ignore it when generating
        # JSON.
        self.__is_deleted = False

    def mark_as_delete(self):
        self.__is_deleted = True

    def is_deleted(self):
        return self.__is_deleted

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

    def __init__(self, fid, doc_name, media_type, language, root_id):
        super().__init__(fid, 'document', None)
        self.doc_name = doc_name
        self.media_type = media_type
        self.language = language
        self.num_sentences = 0
        self.root_id = root_id

    def json_rep(self):
        rep = super().json_rep()
        rep['media_type'] = self.media_type
        rep['language'] = self.language
        rep['num_sentences'] = self.num_sentences
        rep['root'] = self.root_id
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
        self.mutex_fields = set()
        self.interp_score = score

    def get_field(self, name):
        if name in self.__fields:
            return dict(self.__fields.get(name))
        else:
            return None

    def get_field_names(self):
        return self.__fields.keys()

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

    def clear_field(self, name):
        if name in self.__fields:
            self.__fields[name] = defaultdict(dict)

    def add_field(self, name, key_name, content_rep, content,
                  component=None, score=None, multi_value=False,
                  mutex=True):
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
        :param mutex: Whether multiple value of the same key are considered
        mutually exclusive.
        :return:
        """
        # If the key and the corresponding content are the same,
        # then the two interps are consider the same.
        self.__fields[name][key_name][content_rep] = {
            'content': content, 'component': component, 'score': score
        }
        if multi_value:
            self.multi_value_fields.add(name)

        if mutex:
            self.mutex_fields.add(name)

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
                    if field_name in self.mutex_fields:
                        # Multiple interpretation considered as xor.
                        field = {'@type': 'xor', 'args': []}
                    else:
                        field = []

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

                        if field_name in self.mutex_fields:
                            field['args'].append(r)
                        else:
                            field.append(r)
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

    # def add_justification(self, justification):
    #     self.interp.add_field('justification', 'justification',
    #                           justification, justification, multi_value=False)


class RelArgFrame(Frame):
    """
    A frame for the argument of a relation.
    """

    def __init__(self, fid, arg_type, arg_ent_id):
        super().__init__(fid, 'argument', None)

        if ":" in arg_type:
            self.arg_type = arg_type
        else:
            self.arg_type = 'aida:' + arg_type

        self.arg_ent_id = arg_ent_id

    def json_rep(self):
        rep = {
            '@type': 'argument',
            'type': self.arg_type,
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
        self.kv = {}

    def add_value(self, key, value):
        self.kv[key] = value

    def json_rep(self):
        rep = super().json_rep()
        rep['score'] = self.score
        for k, v in self.kv.items():
            rep[k] = v

        return rep


class Span:
    """
    Represent text span (begin and end), according to the reference.
    """

    def __init__(self, reference, begin, length, text):
        self.reference = reference
        self.begin = begin
        self.length = length
        self.text = text

    def json_rep(self):
        return {
            '@type': 'text_span',
            'text': self.text,
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
            length, text, component=None, score=None):
        super().__init__(fid, frame_type, parent, interp_type, component, score)
        if parent:
            self.span = Span(parent.id, begin, length, text)
        else:
            self.span = None

        self.text = text

        self.modifiers = {}

        if parent and parent.type == 'sentence':
            # Inherit key frame from parents.
            self.keyframe = parent.keyframe
        else:
            self.keyframe = None

    def set_text(self, text):
        self.text = text
        self.span.text = text

    def add_modifier(self, modifier_type, modifier_text):
        self.modifiers[modifier_type] = modifier_text

    def json_rep(self):
        rep = super().json_rep()
        if self.span:
            default_provenance = {
                '@type': 'text_span',
                'type': 'aida:base_provenance',
                'reference': self.span.reference,
                'start': self.span.begin,
                'length': self.span.length,
                'text': self.span.text,
                'parent_scope': self.parent.id,
                'modifiers': self.modifiers,
            }

            if self.keyframe:
                default_provenance['keyframe'] = self.keyframe

            rep['provenance'] = default_provenance
        return rep


class Sentence(SpanInterpFrame):
    """
    Represent a sentence.
    """

    def __init__(self, fid, parent, begin, length, text,
                 component=None, keyframe=None, score=None):
        super().__init__(fid, 'sentence', parent, 'sentence_interp',
                         begin, length, text, component=component)
        self.keyframe = keyframe
        self.score = score

    def substring(self, span):
        """
        This substring method take the doc level span, and get the sentence
        substring by correcting the offsets.
        :param span:
        :return:
        """
        begin = span[0] - self.span.begin
        end = span[1] - self.span.begin
        return self.span.text[begin: end]

    def json_rep(self):
        rep = super().json_rep()
        if self.score:
            rep["score"] = self.score
        return rep


class EntityMention(SpanInterpFrame):
    """
    Represent a entity mention (in output, it is called entity_evidence).
    """

    def __init__(self, fid, parent, begin, length, text,
                 component=None, score=None):
        super().__init__(
            fid, 'entity_evidence', parent, 'entity_evidence_interp', begin,
            length, text, component=component, score=score,
        )
        self.entity_types = set()
        self.entity_form = None
        self.salience = None
        self.head_word = None
        self.not_ok = False

    def set_not_ok(self):
        self.not_ok = True

    def set_form(self, entity_form):
        self.entity_form = entity_form

    def get_types(self):
        return list(self.entity_types)

    def add_type(self, ontology, entity_type, score=None, component=None):
        if entity_type == "null":
            return

        full_type = ontology + ":" + entity_type

        if component == self.component:
            # Inherit frame component name.
            component = None

        self.interp.add_field('type', 'type', full_type, full_type,
                              score=score, component=component, mutex=False)
        self.entity_types.add(full_type)

    def add_linking(self, mid, wiki, score, refkbid=None, lang='en',
                    component=None, canonical_name=None):
        if mid:
            fb_link = 'freebase:' + mid
            fb_xref = ValueFrame(None, 'db_reference', score=score,
                                 component=component)
            fb_xref.add_value('id', fb_link)
            self.interp.add_field('xref', 'freebase', mid, fb_xref,
                                  multi_value=True)

        if wiki:
            wiki_link = lang + '_wiki:' + wiki
            wiki_xref = ValueFrame(None, 'db_reference',
                                   score=score, component=component)
            wiki_xref.add_value('id', wiki_link)
            self.interp.add_field('xref', 'wikipedia', wiki, wiki_xref,
                                  component=component, multi_value=True)

        if refkbid:
            refkb_link = 'refkb:' + refkbid
            refkb_xref = ValueFrame(None, 'db_reference',
                                    score=score, component=component)
            refkb_xref.add_value('id', refkb_link)
            refkb_xref.add_value('canonical_name', canonical_name)
            self.interp.add_field('xref', 'refkb', refkbid, refkb_xref,
                                  component=component, multi_value=True)

    def add_salience(self, salience_score):
        self.salience = salience_score

    def json_rep(self):
        # If no type information added.
        if len(self.entity_types) == 0:
            dummy_type = 'conll:MISC'
            self.interp.add_field('type', 'type', dummy_type, dummy_type,
                                  score=0, component=None, mutex=False)

        # Add these interp fields at last, so no extra xor to deal with.
        if self.entity_form:
            self.interp.clear_field('form')
            self.interp.add_field('form', 'form', self.entity_form,
                                  self.entity_form)

        if self.salience:
            self.interp.clear_field('salience')
            self.interp.add_field('salience', 'score', 'score', self.salience)
        return super().json_rep()


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
        self.rel_types = set()
        self.score = score

    def get_types(self):
        return self.rel_types

    def add_type(self, full_type, score=None, component=None):
        if '/' in full_type:
            logging.warning(f"Rejected relation type {full_type}")
            full_type = None

        # Some backward compatibility.
        if ':' not in full_type:
            full_type = "aida:" + full_type

        self.rel_types.add(full_type)

        if component == self.component:
            # Inherit frame component name.
            component = None

        self.interp.add_field(
            'type', 'type', full_type, full_type, component=component,
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

    def __init__(self, arg_role, entity_mention, fid, component=None,
                 score=None):
        super().__init__(fid, 'argument', None, component=component,
                         score=score)
        self.entity_mention = entity_mention
        self.arg_role = arg_role
        self.justification = None

    def json_rep(self):
        rep = super().json_rep()
        rep['type'] = self.arg_role
        rep['text'] = self.entity_mention.text
        rep['arg'] = self.entity_mention.id

        if self.justification:
            rep['justification'] = self.justification

        return rep

    # def add_justification(self, justification):
    #     self.justification = justification


class EventMention(SpanInterpFrame):
    """
    An event mention (in output, it is called event_evidence)
    """

    def __init__(self, fid, parent, begin, length, text, component=None,
                 score=None):
        super().__init__(
            fid, 'event_evidence', parent, 'event_evidence_interp',
            begin, length, text, component=component, score=score,
        )
        # self.trigger = None
        self.event_type = set()
        self.realis = None
        self.arguments = defaultdict(list)
        self.extent_span = None

    # If you call this add_extent function, then the provenance
    # will be the longer extent passed by it.
    def add_extent(self, extent_span):
        self.extent_span = extent_span

    def add_interp(self, full_type, realis, score=None,
                   component=None):
        onto_type, event_type = full_type.split(":")

        if component == self.component:
            # Inherit frame component name.
            component = None

        if full_type is not None:
            self.interp.add_field('type', 'type', onto_type, full_type,
                                  score=score, component=component)
            # Allow multiple event types.
            self.event_type.add(full_type)

        if realis is not None:
            self.interp.add_field('realis', 'realis', realis, realis,
                                  score=score, component=component)
            self.realis = realis

    def get_types(self):
        return list(self.event_type)

    def add_arg(self, ontology, arg_role, entity_mention, arg_id,
                score=None, component=None):
        arg = Argument(ontology + ':' + arg_role, entity_mention, arg_id,
                       component=component, score=score)
        self.arguments[ontology + ':' + arg_role].append(arg)
        return arg

    def add_salience(self, salience_score):
        self.interp.add_field('salience', 'score', 'score', salience_score)

    def pick_args(self, arg_role, alter_args):
        storing_args = []

        def is_good_arg(arg):
            has_type = False
            for t in arg.entity_mention.get_types():
                if not t == 'conll:OTHER':
                    has_type = True
            return has_type or arg.entity_mention.entity_form

        def is_bad_arg(arg):
            if arg.entity_mention.not_ok:
                return True

        if arg_role == 'Internal:Message':
            # Pick longer as message.
            max_len = -1
            longest_arg = None
            for arg in alter_args:
                if len(arg.entity_mention.text) > max_len:
                    longest_arg = arg
                    max_len = len(arg.entity_mention.text)
            if longest_arg:
                storing_args.append(longest_arg)
        else:
            good_args = []
            normal_args = []
            bad_args = []

            for arg in alter_args:
                if is_good_arg(arg):
                    good_args.append(arg)
                elif is_bad_arg(arg):
                    bad_args.append(arg)
                else:
                    normal_args.append(arg)

            if good_args:
                # If there is any good arguments, we will use only those.
                storing_args = good_args
            elif normal_args:
                storing_args = normal_args
            else:
                # Otherwise we have no choice.
                storing_args = bad_args

        return storing_args

    def json_rep(self):
        self.interp.clear_field('args')
        for arg_role, alter_args in self.arguments.items():
            storing_args = self.pick_args(arg_role, alter_args)

            for arg in storing_args:
                self.interp.add_field(
                    'args', arg_role, arg.entity_mention.id, arg,
                    score=arg.score, component=arg.component,
                    multi_value=True
                )
        rep = super().json_rep()

        if self.extent_span:
            extent_provenance = {
                '@type': 'text_span',
                'type': 'aida:extent_provenance',
                'reference': self.extent_span.reference,
                'start': self.extent_span.begin,
                'length': self.extent_span.length,
                'text': self.extent_span.text,
                'parent_scope': self.parent.id,
                'modifiers': self.modifiers,
            }

            if self.keyframe:
                extent_provenance['keyframe'] = self.keyframe

            rep['provenance'] = extent_provenance

        return rep


class CSR:
    """
    Main class that collect and output frames.
    """

    def __init__(self, component_name, run_id, namespace, ontology,
                 media_type='text', onto_check=True):
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
                'ontology': {
                    'prefix': ontology.prefix,
                    'name': ontology.name,
                    'version': ontology.version,
                },
                'runid': 'r%s' % run_id,
            },
        }
        self.onto_check = onto_check

        self._docs = {}

        self.current_doc = None
        self.root_id = None

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

        self.ontology = ontology

    def turnoff_onto_check(self):
        self.onto_check = False

    def turnon_onto_check(self):
        self.onto_check = True

    def set_root(self, root_id):
        self.root_id = root_id
        self.header['meta']['root'] = root_id

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

        def handle_xor(group):
            if isinstance(group, dict):
                return [g['value'] for g in group['args']]
            else:
                return [g['value'] for g in group]

        with open(csr_file) as fin:
            csr_json = json.load(fin)
            media_type = csr_json['meta']['media_type']

            docname = csr_json['meta']['document_id'].replace(
                f"{self.ns_prefix}:", "")

            frame_data = defaultdict(list)

            for frame in csr_json['frames']:
                frame_type = frame['@type']
                fid = frame['@id']

                if frame_type == 'document':
                    self.add_doc(docname, media_type, frame['language'],
                                 docname)
                elif frame_type == 'sentence':
                    start = frame["provenance"]["start"]
                    end = frame["provenance"]["length"] + start

                    self.add_sentence(
                        (start, end), frame["provenance"]['text'], sent_id=fid)
                else:
                    frame_data[frame_type].append(frame)

            entities = {}
            for frame in frame_data['entity_evidence']:
                parent_sent = get_parent_sent(frame)
                interp = frame['interp']
                # onto, t = interp['type'].split(':')
                text = frame["provenance"]['text']
                span = compute_span(frame)
                eid = frame['@id']

                raw_type = interp["type"]

                if isinstance(raw_type, str):
                    entity_types = [raw_type]
                else:
                    entity_types = handle_xor(raw_type)

                for t in entity_types:
                    csr_ent = self.add_entity_mention(
                        span, span, text, t, parent_sent,
                        interp.get('form', None), frame['component'],
                        entity_id=eid
                    )

                    entities[eid] = csr_ent

            for frame in frame_data['event_evidence']:
                parent_sent = get_parent_sent(frame)
                interp = frame['interp']
                # onto, t = interp['type'].split(':')
                text = frame["provenance"]['text']
                span = compute_span(frame)

                raw_type = interp.get("type", None)

                if raw_type is None or isinstance(raw_type, str):
                    event_types = [raw_type]
                else:
                    event_types = handle_xor(raw_type)

                for t in event_types:
                    csr_evm = self.add_event_mention(
                        span, span, text, t, realis=interp.get('realis', None),
                        parent_sent=parent_sent, component=frame['component'],
                        event_id=frame['@id']
                    )

                for mod, v in frame["provenance"]["modifiers"].items():
                    csr_evm.add_modifier(mod, v)

                for arg in interp.get('args', []):
                    arg_values = []
                    if arg['@type'] == 'xor':
                        for sub_arg in arg['args']:
                            arg_values.append(sub_arg)
                    else:
                        arg_values.append(arg)

                    for arg_val in arg_values:
                        if arg_val['@type'] == 'facet':
                            arg_detail = arg_val['value']
                        else:
                            arg_detail = arg_val

                        arg_onto, arg_type = arg_detail['type'].split(':')
                        arg_id = arg_detail['@id']
                        arg_entity_id = arg_detail['arg']
                        ent = entities[arg_entity_id]

                        csr_evm.add_arg(
                            arg_onto, arg_type, ent, arg_id,
                            component=arg_val['component']
                        )

            for frame in frame_data['relation_evidence']:
                fid = frame['@id']
                interp = frame['interp']

                args = [arg['arg'] for arg in interp['args']]
                arg_names = [arg['type'] for arg in interp['args']]

                if "provenance" in frame:
                    span = compute_span(frame)
                    parent_sent = get_parent_sent(frame)

                rel = self.add_relation(
                    args, arg_names,
                    parent_sent=parent_sent,
                    component=frame['component'],
                    relation_id=fid, span=span,
                    score=frame.get('score', None)
                )

                rel.add_type(interp['type'])

    def clear(self):
        self._span_frame_map.clear()
        self._frame_map.clear()
        self._char_sent_map.clear()

    def add_doc(self, doc_name, media_type, language, root_id):
        # ns_docid = self.ns_prefix + ":" + doc_name + "-" + media_type
        ns_docid = self.ns_prefix + ":" + doc_name

        if ns_docid in self._docs:
            return

        doc = Document(ns_docid, doc_name, media_type, language, root_id)
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
                     sent_id=None, score=None):
        if not sent_id:
            sent_id = self.get_id('sent')

        if sent_id in self._frame_map[self.sent_key]:
            return sent_id

        self.current_doc.num_sentences += 1
        sent_text = text if text else ""
        sent = Sentence(
            sent_id, self.current_doc, span[0], span[1] - span[0],
            text=sent_text, component=component, keyframe=keyframe, score=score
        )
        self._frame_map[self.sent_key][sent_id] = sent

        for c in range(span[0], span[1]):
            self._char_sent_map[c] = sent_id

        return sent

    def set_sentence_text(self, sent_id, text):
        self._frame_map[self.sent_key][sent_id].set_text(text)

    def align_to_text(self, span, text, sent):
        """
        Hacking the correct spans.
        :param span: The origin mention span.
        :param text: The origin mention text.
        :param sent: User can specify the hosting sentence.
        :return:
        """
        span = tuple(span)
        if not sent:
            # Find the sentence if not provided.
            sent = self.find_parent_sent(span)

        if not sent:
            logging.warning("No suitable sentence for span {}".format(span))
            return

        fitted_span = self.fit_to_sentence(span, sent)

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
            if "".join(span_text.split()) == "".join(text.split()):
                logging.warning('only white space difference, accepting.')
                return span_text

            logging.warning(
                f"Span text [{span_text}] not aligned with provided [{text}]")
            return None

        return text

    def find_parent_sent(self, span):
        # print('trying on char sent map of size ', len(self._char_sent_map))
        # print('trying with', span, span[0])
        # input('huh')
        if span[0] in self._char_sent_map:
            sentence = self._frame_map[self.sent_key][
                self._char_sent_map[span[0]]]
            return sentence

    def fit_to_sentence(self, span, sentence):
        sent_begin = sentence.span.begin
        sent_end = sentence.span.length + sent_begin

        begin = span[0]
        if begin < sent_begin:
            begin = sent_begin
            logging.warning("Force span {} start at sentence star {} of "
                            "{}".format(span, sent_begin, sentence.id))

        end = span[1]
        if end > sent_end:
            end = sent_end
            logging.warning("Force span {} end at sentence end {} of "
                            "{}".format(span, sent_end, sentence.id))

        return begin, end

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
            if entity_type in conll_to_target:
                return conll_to_target[entity_type]
        return onto_name, entity_type

    def add_relation(self, arguments, arg_names=None, component=None,
                     parent_sent=None, relation_id=None, span=None,
                     score=None):
        """
        Adding a relation mention to CSR. If the span is not provided, it will
        not have a provenance.

        :param parent_sent: The parent sentence of the relation mention.
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

            align_res = self.align_to_text(span, None, parent_sent)
            if not align_res:
                print('no align result')
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

    def add_entity_mention(self, head_span, span, text, entity_type,
                           parent_sent=None, entity_form=None, component=None,
                           entity_id=None, score=None):
        if span is None:
            logging.warning("None span provided for entity")
            return

        if text is None:
            logging.warning("None text provided for entity")
            return

        head_span = tuple(head_span)
        align_res = self.align_to_text(span, text, parent_sent)

        if align_res:
            parent_sent, fitted_span, valid_text = align_res
            sentence_start = parent_sent.span.begin

            if fitted_span in self._span_frame_map[self.entity_key]:
                # Find an entity mention via exact span match.
                entity_mention = self._span_frame_map[self.entity_key][
                    fitted_span]
            elif head_span in self._span_frame_map[self.entity_key + "_head"]:
                # Find an entity mention via head word match.
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
                    entity_mention.set_form(entity_form)
        else:
            return

        if entity_type is not None:
            if '/' not in entity_type:
                ent_onto_type = entity_type.split(':', 1)

                if len(ent_onto_type) == 2:
                    onto = ent_onto_type[0]
                    t = ent_onto_type[1]
                else:
                    # For types without a prefix, add the "aida" prefix.
                    onto = 'aida'
                    t = entity_type

                # This maps from some other ontology to the target ontology.
                onto_type, t = self.map_entity_type(onto, t)
                entity_mention.add_type(onto_type, t, component=component)
            else:
                logging.warning(f"Ignoring entity type {entity_type}")

        return entity_mention

    def add_event_mention(self, head_span, span, text, full_evm_type,
                          realis=None, parent_sent=None, component=None,
                          arg_entity_types=None, event_id=None, score=None,
                          extent_text=None, extent_span=None):
        """
        Add a event mention.
        :param head_span: The span (begin, end) of the head word.
        :param span: The span of the whole mention.
        :param text: The text for this event mention.
        :param full_evm_type: The event type for this event.
        :param realis: The realis status (actual, generic, other) for this event.
        :param parent_sent: The parent sentence containing this mention.
        :param component: The component id that creates this.
        :param arg_entity_types: The set of entity types of this event's arguments.
        :param event_id: The id of this event.
        :param score:  A score for event detection.
        :param extent_text: The extent is the full span of the event, this
        is the extent text.
        :param extent_span: And this is the extent span (begin, end), if both
        the extent_text and extent_span are supplied, the CSR will create an
        extent based provenance instead of the triger based provenance.
        :return:
        """
        if full_evm_type is not None:
            evm_onto_type = full_evm_type.split(':', 1)

            if len(evm_onto_type) == 2:
                onto_name, evm_type = evm_onto_type

                if onto_name == self.ontology.prefix:
                    if self.onto_check:
                        if full_evm_type not in self.ontology.event_onto:
                            logging.warning(
                                f"Event type {full_evm_type} rejected "
                                f"because of ontology name mismatch.")
                            return
            else:
                logging.warning(f"Event type {full_evm_type} rejected "
                                f"because there is no ontology provided.")

        # Annotation on the same span will be reused.
        head_span = tuple(head_span)
        span = tuple(span)

        span_aligned = self.align_to_text(span, text, parent_sent)

        if span_aligned:
            parent_sent, fitted_span, valid_text = span_aligned

            if span in self._span_frame_map[self.event_key]:
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
                    component=component, score=score
                )

                if extent_text and extent_span:
                    extent_aligned = self.align_to_text(
                        extent_span, extent_text, parent_sent)

                    if extent_aligned:
                        e_parent, e_span_fit, e_text_valid = extent_aligned
                        e_relative_begin = e_span_fit[0] - e_parent.span.begin
                        e_length = e_span_fit[1] - e_span_fit[0]
                        extent_span = Span(
                            e_parent.id, e_relative_begin, e_length,
                            e_text_valid)
                        evm.add_extent(extent_span)
                    else:
                        logging.warning(
                            f"Extent [{extent_text}][{extent_span[0]}: "
                            f"{extent_span[1]}] does not aligned well")

                self._frame_map[self.event_key][event_id] = evm

                self._span_frame_map[self.event_key][span] = evm
                self._span_frame_map[self.event_key + "_head"][head_span] = evm
        else:
            return

        if full_evm_type is not None and '/' not in full_evm_type:
            if arg_entity_types:
                full_evm_type = fix_event_type_from_entity(
                    full_evm_type, arg_entity_types
                )
            evm.add_interp(full_evm_type, realis, component=component)

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

    def add_event_arg_by_span(self, evm, arg_head_span, arg_span,
                              arg_text, role, component, score=None):
        ent = self.add_entity_mention(
            arg_head_span, arg_span, arg_text, None, component=component
        )

        if ent:
            arg_id = self.get_id('arg')

            aida_arg_entity_types = []

            for t in ent.get_types():
                if t.startswith('aida:'):
                    aida_arg_entity_types.append(t.split(':')[1])

            arg_onto_name, slot_type = role.split(':', 1)

            arg_onto = None
            if arg_onto_name == 'fn':
                arg_onto = 'framenet'
            elif arg_onto_name == 'pb':
                arg_onto = 'propbank'
            elif arg_onto_name == self.ontology.prefix:
                arg_onto = arg_onto_name
            else:
                return

            return evm.add_arg(arg_onto, slot_type, ent, arg_id,
                               component=component, score=score)

    def add_event_arg(self, evm, ent, ontology, arg_role, component):
        arg_id = self.get_id('arg')
        evm.add_arg(ontology, arg_role, ent, arg_id, component=component)

    def get_json_rep(self):
        rep = {}
        self.header['meta']['document_id'] = self.current_doc.id
        self.header['meta']['root'] = self.current_doc.root_id
        rep.update(self.header)
        rep['frames'] = [self.current_doc.json_rep()]
        for frame_type, frame_info in self._frame_map.items():
            for fid, frame in frame_info.items():
                if not frame.is_deleted():
                    rep['frames'].append(frame.json_rep())
                else:
                    logging.info(
                        f"{fid} ignored since it is marked as deleted.")
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
    if evm_type == 'ldcOnt:Movement.TransportPerson':
        if ('ldcOnt:PER' not in arg_entity_types) and (
                'ldcOnt:VEH.MilitaryVehicle' in arg_entity_types):
            evm_type = 'ldcOnt:Movement.TransportArtifact'
    if evm_type == 'ldcOnt:Personnel.EndPosition':
        if 'ldcOnt:WEA' in arg_entity_types or \
                'ldcOnt:VEH.MilitaryVehicle' in arg_entity_types:
            evm_type = 'ldcOnt:Conflict.Attack'
    return evm_type


if __name__ == '__main__':
    import sys

    # Test loading a csr file
    csr = CSR('Frames_hector_combined', 1, 'data')
    csr.load_from_file(sys.argv[1])
    csr.write(sys.argv[2])

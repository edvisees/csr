import json
from collections import defaultdict
import os
import sys
from event.io.csr import Constants, CSR


def get_span(frame):
    begin = frame['provenance']['start']
    end = frame['provenance']['length'] + begin
    return begin, end


def get_type(frame, default_value):
    t = frame.get('interp', {}).get('type', default_value)

    if t is not None:
        if isinstance(t, str):
            return t
        else:
            t = frame.get('interp', {}).get('type', {}).get('value',
                                                            default_value)
            return t


def get_arg_entity(arg_frame):
    if 'arg' in arg_frame:
        return arg_frame['arg'], arg_frame['type']
    else:
        return arg_frame.get('value', {}).get('arg', None), \
               arg_frame.get('value', {}).get('type', None)


def construct_text(doc_sentences):
    doc_texts = {}
    for docid, content in doc_sentences.items():
        doc_text = ""
        sent_pad = ' '
        for (begin, end), sent, sid in content:
            if begin > len(doc_text):
                pad = sent_pad * (begin - len(doc_text))
                doc_text += pad
            sent_pad = '\n'
            doc_text += sent

        doc_texts[docid] = doc_text + '\n'
    return doc_texts


def strip_ns(name):
    return name.split(':', 1)[1]


def replace_ns(name):
    return '_'.join(reversed(name.split(':', 1)))


def make_text_bound(text_bound_index, t, start, end, text):
    return 'T{}\t{} {} {}\t{}\n'.format(
        text_bound_index,
        t,
        start,
        end,
        text
    )


class CrsConverter:
    def __init__(self):
        self.data = []

    def read_csr(self, path):
        csr = CSR('Read_from_file', 1, 'data')
        csr.load_from_file(path)

        doc_sentences = defaultdict(list)
        event_mentions = defaultdict(dict)
        event_args = defaultdict(list)
        entity_mentions = defaultdict(dict)
        relations = []

        for fid, sent_frame in csr.get_frames(csr.sent_key).items():
            parent = sent_frame.parent
            doc_sentences[parent].append(
                (sent_frame.span.get(), sent_frame.text, sent_frame.id)
            )

        for fid, event_frame in csr.get_frames(csr.event_key).items():
            parent = event_frame.parent
            event_id = event_frame.id
            event_mentions[parent][event_id] = (
                event_frame.span.get(),
                event_frame.event_type,
                event_frame.text
            )

            arg_interps = event_frame.interp.get_field('args')

            if arg_interps:
                for arg_slot, arg_entities in arg_interps.items():
                    for entity_id, arg_entry in arg_entities.items():
                        arg = arg_entry['content']
                        entity_mention = arg.entity_mention
                        full_arg_role = arg.arg_role
                        arg_entity_id = entity_mention.id
                        event_args[event_id].append(
                            (arg_entity_id, full_arg_role)
                        )

        for fid, entity_frame in csr.get_frames(csr.entity_key).items():
            parent = entity_frame.parent
            entity_id = entity_frame.id

            entity_mentions[parent][entity_id] = (
                entity_frame.span.get(),
                entity_frame.entity_types[0],
                entity_frame.text,
            )

        for fid, rel_frame in csr.get_frames(csr.rel_key).items():
            relations.append((rel_frame.relation_type, rel_frame.arguments))

        self.data = doc_sentences, event_mentions, event_args, \
                    entity_mentions, relations

    def write_brat(self, output_dir, keep_onto=False, onto_set=None,
                   extension='.ltf.xml'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        (doc_sentences, event_mentions, event_args, entity_mentions,
         relations) = self.data
        doc_texts = construct_text(doc_sentences)

        for docid, doc_text in doc_texts.items():
            doc_name, media = docid.split('-')
            doc_name = doc_name.replace(extension, '')

            src_output = os.path.join(output_dir, strip_ns(doc_name) + '.txt')
            text_bound_index = 0
            event_index = 0
            with open(src_output, 'w') as out:
                out.write(doc_text)

            ann_output = os.path.join(output_dir, strip_ns(doc_name) + '.ann')

            with open(ann_output, 'w') as out:
                entity2tid = {}

                for sent in doc_sentences[docid]:
                    (sent_start, sent_end), sent_text, sid = sent

                    for entity_id, ent in entity_mentions[sid].items():
                        span, entity_type, text = ent

                        onto, raw_type = entity_type.split(':')

                        if onto_set and onto not in onto_set:
                            full_type = 'OTHER'
                        else:
                            full_type = onto + '_' + raw_type if keep_onto \
                                else raw_type

                        full_type = full_type.replace('.', '_')

                        text_bound = make_text_bound(
                            text_bound_index, full_type,
                            sent_start + span[0], sent_start + span[1],
                            text
                        )

                        entity2tid[entity_id] = "T{}".format(text_bound_index)

                        text_bound_index += 1

                        out.write(text_bound)

                for sent in doc_sentences[docid]:
                    (sent_start, sent_end), sent_text, sid = sent

                    for event_id, evm in event_mentions[sid].items():
                        span, event_type, text = evm

                        onto, raw_type = event_type.split(':')

                        if onto_set and onto not in onto_set:
                            continue

                        full_type = onto + '_' + raw_type if keep_onto \
                            else raw_type

                        full_type = full_type.replace('.', '_')

                        text_bound = make_text_bound(
                            text_bound_index, full_type,
                            sent_start + span[0], sent_start + span[1], text
                        )

                        event_anno = 'E{}\t{}:T{}'.format(
                            event_index, full_type, text_bound_index
                        )

                        if event_id in event_args:
                            args = event_args[event_id]

                            for arg_entity, full_arg_role in args:
                                onto, arg_role = full_arg_role.split(':')

                                if onto_set and onto in onto_set:
                                    arg_role = arg_role.replace('.', '_')
                                else:
                                    arg_role = 'Arg'

                                if arg_entity in entity2tid:
                                    arg_anno = arg_role + ':' + entity2tid[
                                        arg_entity]
                                    event_anno += ' ' + arg_anno

                        event_anno += '\n'

                        text_bound_index += 1
                        event_index += 1

                        out.write(text_bound)
                        out.write(event_anno)


if __name__ == '__main__':
    csr_in, brat_out = sys.argv[1:3]

    converter = CrsConverter()

    for fn in os.listdir(csr_in):
        if fn.endswith('.csr.json'):
            converter.read_csr(os.path.join(csr_in, fn))
            converter.write_brat(brat_out, False, onto_set={'aida'})

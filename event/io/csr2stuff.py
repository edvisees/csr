import json
from collections import defaultdict
import os
import sys
from event.io.csr import Constants, CSR
from event.io.ontology import JsonOntologyLoader


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


class CsrConverter:
    def __init__(self, ontology):
        self.data = []
        self.ontology = ontology

    def read_csr(self, path):
        csr = CSR('Load_from_Disk', 1, 'data', ontology=self.ontology)
        csr.load_from_file(path)
        self.load_csr(csr)

    def load_csr(self, csr):

        doc_sentences = defaultdict(list)
        event_mentions = defaultdict(dict)
        event_args = defaultdict(list)
        entity_mentions = defaultdict(dict)
        relations = []

        for fid, sent_frame in csr.get_frames(csr.sent_key).items():
            parent = sent_frame.parent.id
            doc_sentences[parent].append(
                (sent_frame.span.get(), sent_frame.text, sent_frame.id)
            )

        for fid, event_frame in csr.get_frames(csr.event_key).items():
            parent = event_frame.parent.id
            event_id = event_frame.id
            event_mentions[parent][event_id] = (
                event_frame.span.get(),
                event_frame.get_types(),
                event_frame.text
            )

            for arg_slot, arguments in event_frame.arguments.items():
                for arg in arguments:
                    event_args[event_id].append(
                        (
                            arg.entity_mention.id,
                            arg.arg_role
                        )
                    )

        for fid, entity_frame in csr.get_frames(csr.entity_key).items():
            parent = entity_frame.parent.id
            entity_id = entity_frame.id

            entity_mentions[parent][entity_id] = (
                entity_frame.span.get(),
                entity_frame.get_types(),
                entity_frame.text,
            )

        for fid, rel_frame in csr.get_frames(csr.rel_key).items():
            for rt in rel_frame.get_types():
                relations.append((rt, rel_frame.arguments))

        self.data = (doc_sentences, event_mentions, event_args,
                     entity_mentions, relations)

    def write_brat(self, output_dir, keep_onto=False, onto_set=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        (doc_sentences, event_mentions, event_args, entity_mentions,
         relations) = self.data
        doc_texts = construct_text(doc_sentences)

        for docid, doc_text in doc_texts.items():
            src_output = os.path.join(output_dir, strip_ns(docid) + '.txt')
            text_bound_index = 0
            event_index = 0
            with open(src_output, 'w') as out:
                out.write(doc_text)

            print(f'Doc {docid} contains '
                  f'{len(doc_sentences[docid])} sentences, '
                  f'{len(event_mentions)} events, '
                  f'{len(entity_mentions)} entities, '
                  f'{len(event_args)} event args, '
                  f'{len(relations)} relations.'
                  )

            ann_output = os.path.join(output_dir, strip_ns(docid) + '.ann')

            with open(ann_output, 'w') as out:
                entity2tid = {}

                for sent in doc_sentences[docid]:
                    (sent_start, sent_end), sent_text, sid = sent

                    for entity_id, ent in entity_mentions[sid].items():
                        span, entity_types, text = ent

                        for entity_type in entity_types:
                            onto, raw_type = entity_type.split(':')

                            if onto_set and onto not in onto_set:
                                # print(entity_type, 'not in ontology')
                                # full_type = 'OTHER'
                                continue
                            else:
                                full_type = onto + '_' + raw_type if keep_onto \
                                    else raw_type

                            full_type = full_type.replace('.', '_')

                            text_bound = make_text_bound(
                                text_bound_index, full_type,
                                sent_start + span[0], sent_start + span[1],
                                text
                            )

                            entity2tid[entity_id] = "T{}".format(
                                text_bound_index)

                            text_bound_index += 1

                            out.write(text_bound)

                for sent in doc_sentences[docid]:
                    (sent_start, sent_end), sent_text, sid = sent

                    for event_id, evm in event_mentions[sid].items():
                        span, event_types, text = evm

                        for event_type in event_types:
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
                                        # Ignore out of ontology arguments.
                                        # arg_role = 'Arg'
                                        continue

                                    if arg_entity in entity2tid:
                                        arg_anno = arg_role + ':' + entity2tid[
                                            arg_entity]
                                        event_anno += ' ' + arg_anno

                            event_anno += '\n'

                            text_bound_index += 1
                            event_index += 1

                            out.write(text_bound)
                            out.write(event_anno)

                r_id = 1
                for rel_type, rel_args in relations:
                    # Like this  "R1	Origin Arg1:T3 Arg2:T4"
                    if 'coreference' in rel_type:
                        continue

                    onto, raw_type = rel_type.split(':')
                    if onto not in onto_set:
                        continue

                    r_str = f'R{r_id}\t' + raw_type
                    r_id += 1

                    i = 0
                    for rel_arg in rel_args:
                        i += 1
                        arg_name = f'Arg{i}'
                        arg_type = rel_arg.arg_type.split(':')[1]
                        r_str += ' ' + arg_name + ':' + entity2tid[
                            rel_arg.arg_ent_id]

                    out.write(r_str + '\n')


if __name__ == '__main__':
    csr_in, brat_out, onto_path = sys.argv[1:4]
    ontology = JsonOntologyLoader(onto_path)

    converter = CsrConverter(ontology)

    print("Input directory is ", csr_in)
    print("Brat output directory is ", brat_out)

    for fn in os.listdir(csr_in):
        if fn.endswith('.csr.json'):
            converter.read_csr(os.path.join(csr_in, fn))
            converter.write_brat(brat_out, False, onto_set={'ldcOnt'})

"""
A small script to convert AIDA annotations to CSR
"""

import sys
import os
from event.io.ontology import JsonOntologyLoader
from event.mention.combine_runner import read_parent_child_info, read_source
from traitlets.config.loader import PyFileConfigLoader
from traitlets.config import Configurable
from event import util
import logging
from traitlets import (
    Unicode,
    Bool,
)
from event.io.csr2stuff import CsrConverter


def tab_to_field(tab_file):
    header = []
    content = {}
    with open(tab_file) as t:
        for l in t:
            fields = l.strip().split('\t')
            if len(header) == 0:
                header = fields
                if 'root_uid' not in header:
                    return None
            else:
                root_uid = fields[header.index('root_uid')]
                try:
                    content[root_uid].append(fields)
                except KeyError:
                    content[root_uid] = [fields]
    return header, content


def main():
    anno_data = {}
    for f in os.listdir(config.aida_annotation):
        data = tab_to_field(os.path.join(config.aida_annotation, f))
        if f.endswith('_evt_mentions.tab'):
            anno_data['event_mention'] = data
        elif f.endswith('_evt_slots.tab'):
            anno_data['event_slots'] = data
        elif f.endswith('_arg_mentions.tab'):
            anno_data['event_args'] = data
        elif f.endswith('_rel_mentions.tab'):
            anno_data['relation_mention'] = data
        elif f.endswith('_rel_slots.tab'):
            anno_data['relation_slot'] = data

    # Create a CSR
    ontology = JsonOntologyLoader(config.ontology_path)
    child2root = read_parent_child_info(config.parent_children_tab)
    converter = CsrConverter(ontology)

    component = 'ldc_gold_standard'

    for csr, docid in read_source(config.source_folder, config.language,
                                  ontology, child2root, component):
        logging.info('Working with docid: {}'.format(docid))

        # Switch off ontology check.
        csr.turnoff_onto_check()
        root_uid = child2root[docid]

        evm_header, evm_content = anno_data['event_mention']
        arg_header, arg_content = anno_data['event_args']
        eslot_header, eslot_content = anno_data['event_slots']

        relm_header, relm_content = anno_data['relation_mention']
        rels_header, rels_content = anno_data['relation_slot']

        csr_args = {}

        # Add events.
        if root_uid in evm_content:
            event_data = evm_content[root_uid]
            arg_data = arg_content[root_uid]
            slot_data = eslot_content[root_uid]

            csr_events = {}
            for evm in event_data:
                if evm[evm_header.index('textoffset_startchar')].startswith('EMPTY_'):
                    continue

                span = (int(evm[evm_header.index('textoffset_startchar')]) - 1,
                        int(evm[evm_header.index('textoffset_endchar')]))
                # TODO: fix this properly using ltf_span_style/off_start_by_1
                span = (span[0] + 1, span[1] + 1)
                text = evm[evm_header.index('text_string')]
                full_type = 'ldcOnt:' + evm[evm_header.index('type')] + '_' \
                            + evm[evm_header.index('subtype')] + '_' \
                            + evm[evm_header.index('subsubtype')]
                csr_evm = csr.add_event_mention(
                    span, span, text, full_type, component=component,
                )

                if csr_evm:
                    csr_events[
                        evm[evm_header.index('eventmention_id')]] = csr_evm

            for arg in arg_data:
                if arg[arg_header.index('textoffset_startchar')].startswith('EMPTY_'):
                    continue

                span = (int(arg[arg_header.index('textoffset_startchar')]) - 1,
                        int(arg[arg_header.index('textoffset_endchar')]))
                # TODO: fix this properly using ltf_span_style/off_start_by_1
                span = (span[0] + 1, span[1] + 1)
                text = arg[arg_header.index('text_string')]

                full_type = 'ldcOnt:' + arg[arg_header.index('type')] + '_' \
                            + arg[arg_header.index('subtype')] + '_' \
                            + arg[arg_header.index('subsubtype')]

                ent = csr.add_entity_mention(
                    span, span, text, full_type, component=component
                )

                if ent:
                    csr_args[arg[arg_header.index('argmention_id')]] = ent

            slot_id = 0
            for slot in slot_data:
                eid = slot[eslot_header.index('eventmention_id')]
                slot_type = slot[eslot_header.index('slot_type')]
                aid = slot[eslot_header.index('argmention_id')]

                real_slot_type = slot_type[11:]

                if eid in csr_events and aid in csr_args:
                    e = csr_events[eid]
                    a = csr_args[aid]
                    e.add_arg('ldcOnt', real_slot_type, a, f'slot_{slot_id}',
                              component=component)
                    slot_id += 1

        if root_uid in rels_content:
            rel_slots = rels_content[root_uid]
            rel_mentions = relm_content[root_uid]

            csr_relms = {}
            for relm in rel_mentions:
                if relm[relm_header.index(
                        'textoffset_startchar')].startswith('EMPTY_'):
                    continue

                span = (
                    int(relm[relm_header.index('textoffset_startchar')]) - 1,
                    int(relm[relm_header.index('textoffset_endchar')])
                )
                # TODO: fix this properly using ltf_span_style/off_start_by_1
                span = (span[0] + 1, span[1] + 1)
                csr_rel = csr.add_relation([], component='corenlp', span=span)

                full_type = 'ldcOnt:' + relm[relm_header.index('type')] + '_' \
                            + relm[relm_header.index('subtype')] + '_' \
                            + relm[relm_header.index('subsubtype')]

                if csr_rel:
                    csr_relms[relm[
                        relm_header.index('relationmention_id')]] = csr_rel
                    csr_rel.add_type(full_type)

            for rel_slot in rel_slots:
                rel_m_id = rel_slot[rels_header.index('relationmention_id')]
                if rel_m_id in csr_relms:
                    rel_m = csr_relms[rel_m_id]

                    rel_m_type = rel_slot[rels_header.index('slot_type')]
                    real_rel_type = 'ldcOnt:' + rel_m_type[11:]

                    aid = rel_slot[rels_header.index('argmention_id')]

                    if aid in csr_args:
                        rel_m.add_named_arg(real_rel_type, csr_args[aid].id)

        converter.load_csr(csr)
        converter.write_brat(config.brat_output_path, False,
                             onto_set={'ldcOnt'})

        # input('check doc ' + docid)


if __name__ == '__main__':
    class AidaConvertParams(Configurable):
        aida_annotation = Unicode(help='AIDA annotation directory.').tag(
            config=True)
        ontology_path = Unicode(help='Ontology url or path.').tag(config=True)
        parent_children_tab = Unicode(
            help='File parent children relations provided').tag(config=True)
        source_folder = Unicode(help='source text folder').tag(config=True)
        language = Unicode(help='Language of data.').tag(config=True)
        brat_output_path = Unicode(help='Path to store the brat output.').tag(
            config=True)


    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    cl_conf = util.load_command_line_config(sys.argv[2:])
    conf.merge(cl_conf)
    config = AidaConvertParams(config=conf)

    util.set_basic_log()
    main()

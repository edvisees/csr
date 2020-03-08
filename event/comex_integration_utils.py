import sys
import traceback
import json


def get_frames(data, frame_type):
    ret = []
    for frame in data['frames']:
        if frame['@type'] == frame_type:
            ret.append(frame)
    return ret


def collect_modifiers(comex_ent_frame, ent_frame):
    modifiers = comex_ent_frame['provenance']['modifiers']
    for modifier in ['NEG', 'MOD']:
        text = modifiers.get(modifier)
        if text is not None:
            ent_frame.add_modifier(modifier_type=modifier, modifier_text=text)


def add_comex(comex_file, csr):
    component_name = 'COMEX'
    with open(comex_file) as comex:
        # Hector warned us agains using CSR loader code...
        data = json.load(comex)
        frame_indexer = {}
        # Entities
        comex_ent_frames = get_frames(data, 'entity_evidence')
        for comex_ent_frame in comex_ent_frames:
            try:
                start = int(comex_ent_frame['interp']['span_for_combined_start'])
                end = int(comex_ent_frame['interp']['span_for_combined_end'])
                span = [start, end]
                text = comex_ent_frame['provenance']['text']
                entity_type = comex_ent_frame['interp']['type']
                form = comex_ent_frame['interp']['form']
                head_span_start = int(comex_ent_frame['interp']['head_span_start'])
                head_span_end = int(comex_ent_frame['interp']['head_span_end'])
                head_span = [head_span_start, head_span_end]
                concept_name = comex_ent_frame['interp']['concept']
                ent_id = comex_ent_frame['@id']
                score = comex_ent_frame['interp']['score']

                ent_frame = csr.add_entity_mention(head_span, span, text, entity_type,
                                                   parent_sent=None, entity_form=form,
                                                   component=component_name,
                                                   entity_id=None, score=score)
                if ent_frame:
                    frame_indexer[ent_id] = ent_frame.id
                    # adding concepts into CSR
                    ent_frame.interp.add_field(name='concept', key_name='concept',
                                               content=concept_name, content_rep=concept_name,
                                               component=component_name)
                    if comex_ent_frame['interp'].get('fringe') is not None:
                        fringe_name = comex_ent_frame['interp'].get('fringe')
                        ent_frame.interp.add_field(name='fringe', key_name='fringe',
                                                   content=fringe_name, content_rep=fringe_name,
                                                   component=component_name)
                    if comex_ent_frame['interp'].get('xref') is not None:
                        xref_list = comex_ent_frame['interp'].get('xref')
                        for xref in xref_list:
                            component = xref['component']
                            score = xref['score']
                            canonical_name = xref['canonical_name']
                            if len(xref['id'].split(':')) != 2:
                                continue
                            prefix, kbid = xref['id'].split(':')
                            refkbid = kbid if prefix == 'refkb' else None
                            comexkbid = kbid if prefix == 'comexkb' else None
                            ent_frame.add_linking(mid=None, wiki=None, score=score, refkbid=refkbid,
                                                  component=component,
                                                  canonical_name=canonical_name, comexkbid=comexkbid)
                    collect_modifiers(comex_ent_frame, ent_frame)
            except Exception:
                sys.stderr.write("ERROR: Exception occurred while processing file {0}\n".format(comex_file))
                sys.stderr.write(json.dumps(comex_ent_frame, indent=4))
                traceback.print_exc()

        # Events
        comex_ev_frames = get_frames(data, 'event_evidence')
        for comex_ev_frame in comex_ev_frames:
            span_start = int(comex_ev_frame['interp']['regular_span_start'])
            span_end = int(comex_ev_frame['interp']['regular_span_end'])
            span = [span_start, span_end]
            extent_span_start = int(comex_ev_frame['interp']['extent_span_start'])
            extent_span_end = int(comex_ev_frame['interp']['extent_span_end'])
            extent_span = [extent_span_start, extent_span_end]
            head_span_start = int(comex_ev_frame['interp']['head_span_start'])
            head_span_end = int(comex_ev_frame['interp']['head_span_end'])
            head_span = [head_span_start, head_span_end]
            text = comex_ev_frame['interp']['regular_text']
            extent_text = comex_ev_frame['provenance']['text']
            event_type = comex_ev_frame['interp']['type']
            score = comex_ev_frame['interp']['score']

            ev_frame = csr.add_event_mention(head_span, span, text, event_type,
                                             realis=None, parent_sent=None,
                                             component=component_name,
                                             arg_entity_types=None, event_id=None, score=score,
                                             extent_span=extent_span, extent_text=extent_text)
            if ev_frame:
                ev_id = comex_ev_frame['@id']
                frame_indexer[ev_id] = ev_frame.id
                collect_modifiers(comex_ev_frame, ev_frame)

                # Basically, for arg in csr for the event
                # 1. get_frame
                # 2. csr.add_event_arg
                if 'args' not in comex_ev_frame['interp'].keys():
                    continue
                for event_arg in comex_ev_frame['interp']['args']:
                    if event_arg['@type'] == 'xor':
                        arg_list = [facet['value'] for facet in event_arg['args']]
                    else:
                        arg_list = [event_arg]
                    for arg in arg_list:
                        ontology_prefix = arg['type'].split(':')[0]
                        arg_role_no_prefix = arg['type'].split(':')[1]

                        # The arg is not found in the previous entity stage
                        if arg['arg'] not in frame_indexer.keys():
                            continue
                        arg_ent_frame_id = frame_indexer[arg['arg']]
                        arg_ent_frame = csr.get_frame(csr.entity_key, arg_ent_frame_id)
                        csr.add_event_arg(ev_frame, arg_ent_frame, ontology=ontology_prefix,
                                          arg_role=arg_role_no_prefix,
                                          component=component_name)
        # Relations
        comex_rel_frames = get_frames(data, 'relation_evidence')
        for comex_rel_frame in comex_rel_frames:
            if comex_rel_frame['interp']['type'] == 'aida:entity_coreference':
                continue
            start = int(comex_rel_frame['interp']['span_start'])
            end = int(comex_rel_frame['interp']['span_end'])
            span = [start, end]
            arg_ids = []
            arg_roles = []
            for rel_arg in comex_rel_frame['interp']['args']:
                if rel_arg['arg'] not in frame_indexer.keys():
                    continue
                arg_roles.append(rel_arg['type'])
                arg_ent_frame_id = frame_indexer[rel_arg['arg']]
                arg_ids.append(arg_ent_frame_id)

            rel_frame = csr.add_relation(arg_ids, arg_names=arg_roles,
                                         component=component_name,
                                         relation_id=None, span=span, score=1.0)
            if rel_frame:
                rel_id = comex_rel_frame['@id']
                frame_indexer[rel_id] = rel_frame.id

                rel_type = comex_rel_frame['interp']['rel_type']
                rel_frame.add_type(rel_type)
                collect_modifiers(comex_rel_frame, rel_frame)



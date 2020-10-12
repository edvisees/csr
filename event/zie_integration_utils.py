#

# writing from zie's json format into the csr

import json
import logging
import math
from collections import Counter

# helper function for add ef
def add_ef(csr, arg_ef, check_type=False):
    extra_info = arg_ef.get("extra_info", {})
    span_info = extra_info.get("posi")
    if span_info is not None:
        ef_type = arg_ef["type"]
        if check_type and ef_type.split(":")[0] not in ["ldcOnt", "aida"]:
            ef_type = None
        ef_score = arg_ef.get("score")
        if ef_score is not None and ef_score < 0.:
            ef_score = math.exp(ef_score)
        ent = csr.add_entity_mention(
            span_info["head_span"], span_info["span"], span_info["text"], ef_type,
            entity_form=extra_info.get("entity_form"), component=extra_info.get("component"), score=ef_score,
        )
        return ent
    else:
        return None

# read file and add to csr
def add_zie_event(zie_event_file, csr):
    with open(zie_event_file) as fd:
        # read json
        doc = json.load(fd)
        evt_posi_mapping = Counter()
        # --
        # add (certain) efs
        ef_all_dict_mappings = {}  # ef-id -> ef (all)
        ef_dict_mappings = {}  # ef-id -> ef
        ef_csr_mappings = {}  # ef-id -> csr_ef
        for ef in doc["entity_mentions"] + doc["fillers"]:
            ef_all_dict_mappings[ef["id"]] = ef
            if ef["type"].startswith("ldcOnt:") or ef["type"].startswith("aida:"):
                ef_dict_mappings[ef["id"]] = ef  # store for later adding
        # --
        # add events
        evt_csr_mappings = {}  # event-id -> csr_evm
        for evt in doc["event_mentions"]:
            if evt["type"].startswith("rel:"):
                continue  # these will be added later
            if not (evt["type"].startswith("ldcOnt:") or evt["type"].startswith("aida:")):
                logging.info(f"Skipping evt for {evt['type']}({evt['extra_info']['posi']['text']})")
                continue
            extra_info = evt["extra_info"]
            span_info = extra_info["posi"]
            # add one
            csr_evm = csr.add_event_mention(
                span_info["head_span"], span_info["span"], span_info["text"], evt["type"],
                realis=evt["realis"], component=extra_info["component"], score=math.exp(evt["score"]),
                extent_text=span_info.get('extra_text', None), extent_span=span_info.get('extra_span', None)
            )
            # span_key = tuple(span_info["span"])
            # evt_posi_mapping[span_key] += 1
            # if evt_posi_mapping[span_key] > 1:
            #     logging.warning("Same posi-span evt inside zie!!")
            # add args
            if csr_evm is not None:
                evt_csr_mappings[evt["id"]] = csr_evm
            else:
                logging.warning(f"Adding evt failed for {evt['type']}({span_info['text']})")
        # --
        # add args
        for evt in doc["event_mentions"]:
            csr_evm = evt_csr_mappings.get(evt["id"])
            if csr_evm is None:
                continue  # adding failed
            for arg in evt["em_arg"]:
                if not (arg["role"].startswith("ldcOnt:") or arg["role"].startswith("aida:")):
                    logging.info(f"Skipping evt arg for {arg['role']}({arg['extra_info']['posi']['text']})")
                    continue
                # do we want to add things that might not be in ontology?
                # if not arg["role"].startswith("ldcOnt:"):
                #     continue
                arg_extra_info = arg["extra_info"]
                arg_span_info = arg_extra_info["posi"]
                if arg_extra_info.get("is_evt2evt_link", False):
                    # directly use event's adding method to add another event as arg
                    arg_evm = evt_csr_mappings.get(arg["aid"])
                    if arg_evm is not None:
                        arg_onto, slot_type = arg["role"].split(":", 1)
                        arg_id = csr.get_id('arg')
                        csr_arg = csr_evm.add_arg(arg_onto, slot_type, arg_evm, arg_id,
                                                  component=arg_extra_info["component"], score=math.exp(arg["score"]))
                    else:
                        csr_arg = None
                else:
                    # add ent only if used as arg
                    arg_ef = ef_dict_mappings.get(arg["aid"])
                    if arg_ef is not None:
                        if arg["aid"] not in ef_csr_mappings:  # no repeated adding
                            ent = add_ef(csr, arg_ef)
                            if ent is not None:  # set for later use
                                ef_csr_mappings[arg["aid"]] = ent
                    # add arg
                    csr_arg = csr.add_event_arg_by_span(
                        csr_evm, arg_span_info["head_span"], arg_span_info["span"], arg_span_info["text"], arg["role"],
                        component=arg_extra_info["component"], score=math.exp(arg["score"])
                    )
                if csr_arg is None:
                    logging.warning(f"Adding evt arg failed for {arg['role']}({arg_span_info['text']})")
        # --
        # add special rels
        for rel in doc["event_mentions"]:
            if not rel["type"].startswith("rel:"):
                continue
            try:  # to be safe ...
                extra_info = rel["extra_info"]
                span_info = extra_info["posi"]
                rel_type = "ldcOnt:" + rel["type"].split(":")[-1]
                rel_arg_ids, rel_arg_roles = [], []
                for arg in rel["em_arg"]:
                    if not (arg["role"].startswith("rel:") or arg["role"].startswith("aida:")):
                        continue
                    # find the arg
                    arg_extra_info = arg["extra_info"]
                    if arg_extra_info.get("is_evt2evt_link", False):
                        one_arg = evt_csr_mappings.get(arg["aid"])
                    else:
                        arg_ef = ef_all_dict_mappings.get(arg["aid"])
                        one_arg = ef_csr_mappings.get(arg["aid"])
                        if one_arg is None and arg_ef is not None:
                            one_arg = add_ef(csr, arg_ef, check_type=True)
                            if one_arg is not None:
                                ef_csr_mappings[arg["aid"]] = one_arg
                    # skip if not added
                    if one_arg is None:
                        logging.warning(f"Adding rel_arg failed for {arg})")
                        continue
                    # put it
                    arg_role = arg['role'] if arg["role"].startswith("aida:") else f"{rel_type}_{arg['role'].split(':')[-1]}"
                    rel_arg_ids.append(one_arg.id)
                    rel_arg_roles.append(arg_role)
                # finally checking
                if len(rel_arg_ids) >= 2:  # allow extra ones for time
                    # finally adding
                    csr_rel = csr.add_relation(
                        arguments=rel_arg_ids, arg_names=rel_arg_roles,
                        component=extra_info.get("component"), span=span_info["span"],
                        score=math.exp(rel["score"]),
                    )
                    if csr_rel:
                        csr_rel.add_type(rel_type)
                else:
                    logging.warning(f"Adding rel failed (invalid num of args) for {rel})")
            except:
                logging.warning(f"Adding rel failed (some except?) for {rel})")
        # --

# =====
# possible ways to modify "combine_runner.py"
"""
# adding a config option for the dir of zie_event: config.zie_event
# after "if config.conllu_folder:" and before "if config.rich_event:", insert these
# =====
        if config.zie_event:
            if os.path.exists(config.zie_event):
                zie_event_file = find_by_id(config.zie_event, docid)
                if zie_event_file:
                    logging.info("Adding events with zie output: {}".format(zie_event_file))
                    zie_integration_utils.add_zie_event(zie_event_file, csr)
                else:
                    logging.error(f"Cannot find zie output for {docid}")
            else:
                logging.info("No zie event output.")
# =====
"""

# =====
# how to move these to _export?
# msp/ -> msp/; tasks/zie/{common,main,models2,models3,opera,tools} -> tasks/zie/; tasks/zie/csr/* -> tasks/zie/csr/(setup)
# fix previous eval bug
# sed -i 's/ldcOnt:PER:/ldcOnt:PER./g' *
# run combine_runner: at csr dir
# output_base=../hector_out/english/ csr_resources=../csr_resources/ PYTHONPATH=./ python event/mention/combine_runner.py conf/aida/combine.py  --CombineParams.csr_output=../zout_csr_norich/ --CombineParams.rich_event=NOPE
# output_base=../hector_out/english/ csr_resources=../csr_resources/ PYTHONPATH=./ python event/mention/combine_runner.py conf/aida/combine.py  --CombineParams.csr_output=../zout_csr_rich/
# visualization
# python -m event.io.csr2stuff ../zout_csr_norich/ ../zout_visual_norich/ ../csr_resources/LDCOntologyM36.jsonld

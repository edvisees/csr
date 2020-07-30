#

# writing from zie's json format into the csr

import json
import logging
import math
from collections import Counter

# read file and add to csr
def add_zie_event(zie_event_file, csr):
    with open(zie_event_file) as fd:
        # read json
        doc = json.load(fd)
        evt_posi_mapping = Counter()
        # add (certain) efs
        for ef in doc["entity_mentions"] + doc["fillers"]:
            if ef["type"].startswith("ldcOnt:") or ef["type"].startswith("aida:"):
                extra_info = ef.get("extra_info", {})
                span_info = extra_info.get("posi")
                if span_info is not None:
                    ent = csr.add_entity_mention(
                        span_info["head_span"], span_info["span"], span_info["text"], ef["type"],
                        component=extra_info["component"], score=math.exp(ef["score"])
                    )
        # add events
        evt_mappings = {}  # event-id -> csr_evm
        for evt in doc["event_mentions"]:
            extra_info = evt["extra_info"]
            span_info = extra_info["posi"]
            # add one
            csr_evm = csr.add_event_mention(
                span_info["head_span"], span_info["span"], span_info["text"], evt["type"],
                realis=evt["realis"], component=extra_info["component"], score=math.exp(evt["score"]),
                extent_text=span_info.get('extra_text', None), extent_span=span_info.get('extra_span', None)
            )
            span_key = tuple(span_info["span"])
            evt_posi_mapping[span_key] += 1
            if evt_posi_mapping[span_key] > 1:
                logging.warning("Same posi-span evt inside zie!!")
            # add args
            if csr_evm is not None:
                evt_mappings[evt["id"]] = csr_evm
            else:
                logging.warning(f"Adding evt failed for {evt['type']}({span_info['text']})")
        # add args
        for evt in doc["event_mentions"]:
            csr_evm = evt_mappings.get(evt["id"])
            if csr_evm is None:
                continue  # adding failed
            for arg in evt["em_arg"]:
                # do we want to add things that might not be in ontology?
                # if not arg["role"].startswith("ldcOnt:"):
                #     continue
                arg_extra_info = arg["extra_info"]
                arg_span_info = arg_extra_info["posi"]
                if arg_extra_info.get("is_evt2evt_link", False):
                    # directly use event's adding method to add another event as arg
                    arg_evm = evt_mappings.get(arg["aid"])
                    if arg_evm is not None:
                        arg_onto, slot_type = arg["role"].split(":", 1)
                        arg_id = csr.get_id('arg')
                        csr_arg = csr_evm.add_arg(arg_onto, slot_type, arg_evm, arg_id,
                                                  component=arg_extra_info["component"], score=math.exp(arg["score"]))
                    else:
                        csr_arg = None
                else:
                    csr_arg = csr.add_event_arg_by_span(
                        csr_evm, arg_span_info["head_span"], arg_span_info["span"], arg_span_info["text"], arg["role"],
                        component=arg_extra_info["component"], score=math.exp(arg["score"])
                    )
                if not csr_arg:
                    logging.warning(f"Adding evt arg failed for {arg['role']}({arg_span_info['text']})")

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

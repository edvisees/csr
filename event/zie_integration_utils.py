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
        evt_mappings = {}  # for debugging
        # add events
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
            # # ----- for debugging
            # if id(csr_evm) not in evt_mappings:
            #     evt_mappings[id(csr_evm)] = []
            # evt_mappings[id(csr_evm)].append(evt)
            # if len(evt_mappings[id(csr_evm)]) > 1:
            #     import pdb
            #     pdb.set_trace()
            # # -----
            # add args
            if csr_evm:
                for arg in evt["em_arg"]:
                    arg_extra_info = arg["extra_info"]
                    arg_span_info = arg_extra_info["posi"]
                    csr_arg = csr.add_event_arg_by_span(
                        csr_evm, arg_span_info["head_span"], arg_span_info["span"], arg_span_info["text"], arg["role"],
                        component=arg_extra_info["component"], score=math.exp(arg["score"])
                    )
                    if not csr_arg:
                        logging.warning(f"Adding evt arg failed for {arg['role']}({arg_span_info['text']})")
            else:
                logging.warning(f"Adding evt failed for {evt['type']}({span_info['text']})")

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
# python -m event.io.csr2stuff ../zout_csr_norich/ ../zout_visual_norich/ ../csr_resources/LDCOntology_v0.1.jsonld

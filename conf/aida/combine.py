import os
from event.util import get_env

output_base = get_env('output_base')

if 'combine_language' in os.environ:
    c.DetectionParams.language = get_env('combine_language')
else:
    c.DetectionParams.language = 'en'

c.DetectionParams.resource_folder = get_env('csr_resources')
# '/home/zhengzhl/workspace/resources/fndata-1.7/frame'
c.DetectionParams.frame_lexicon = get_env('frame_lex')

c.DetectionParams.word_embedding_dim = 300
c.DetectionParams.position_embedding_dim = 50
c.DetectionParams.dropout = 0.5
c.DetectionParams.input_format = 'conllu'
c.DetectionParams.no_punct = True
c.DetectionParams.model_name = 'frame_rule'

# '/home/zhengzhl/workspace/aida/domain_frames/'
c.DetectionParams.resource_folder = get_env('csr_resources')

c.DetectionParams.tag_list = os.path.join(c.DetectionParams.resource_folder,
                                          'target_frames.txt')
c.DetectionParams.no_sentence = True

c.CombineParams.ontology_path = os.path.join(
    c.DetectionParams.resource_folder, 'LDCOntology_v0.1.jsonld'
)
c.CombineParams.output_folder = output_base
c.CombineParams.conllu_folder = os.path.join(output_base, 'conllu')
c.CombineParams.csr_output = os.path.join(output_base, 'csr')
c.CombineParams.rich_event = os.path.join(output_base, 'rich',
                                          'simple_run_mapped')
c.CombineParams.edl_json = os.path.join(output_base, 'entity')
c.CombineParams.relation_json = os.path.join(output_base, 'relations')
c.CombineParams.salience_data = os.path.join(output_base, 'salience')
c.CombineParams.dbpedia_wiki_json = os.path.join(output_base, 'wiki')
c.CombineParams.source_folder = os.path.join(output_base, 'txt')
c.CombineParams.add_rule_detector = True

# This is the parent child from the eval source.
if 'parent_child_tab' in os.environ:
    print(
        "Env variable [parent_child_tab] not provided, will not write parent"
        " information in CSR.")
    c.CombineParams.parent_children_tab = get_env('parent_child_tab')

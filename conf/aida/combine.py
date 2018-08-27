import os
from event.util import get_env

output_base = get_env('output_base')

# '/home/zhengzhl/workspace/resources/fndata-1.7/frame'
c.DetectionParams.frame_lexicon = get_env('frame_lex')

# '/home/zhengzhl/workspace/aida/domain_frames/'
c.DetectionParams.resource_folder = get_env('domain_frames')

c.DetectionParams.word_embedding_dim = 300
c.DetectionParams.position_embedding_dim = 50
c.DetectionParams.dropout = 0.5
c.DetectionParams.input_format = 'conllu'
c.DetectionParams.no_punct = True
c.DetectionParams.model_name = 'frame_rule'

c.DetectionParams.event_list = os.path.join(c.DetectionParams.resource_folder,
                                            'ontology_event_list.txt')
c.DetectionParams.entity_list = os.path.join(c.DetectionParams.resource_folder,
                                             'ontology_entity_list.txt')
c.DetectionParams.relation_list = os.path.join(
    c.DetectionParams.resource_folder, 'ontology_frames.txt')
c.DetectionParams.tag_list = os.path.join(c.DetectionParams.resource_folder,
                                          'target_frames.txt')
c.DetectionParams.ontology_path = 'resources/SeedlingOntology'
c.DetectionParams.seedling_event_mapping = 'resources/' \
                                           'seedling_event_mapping.txt'
c.DetectionParams.seedling_argument_mapping = 'resources/' \
                                              'seedling_argument_mapping.txt'
c.DetectionParams.no_sentence = True
c.DetectionParams.language = 'en'

c.CombineParams.output_folder = output_base
c.CombineParams.conllu_folder = os.path.join(output_base, 'conllu')
c.CombineParams.csr_output = os.path.join(output_base, 'csr')
c.CombineParams.rich_event = os.path.join(output_base, 'rich', 'simple_run')
c.CombineParams.edl_json = os.path.join(output_base, 'entity')
c.CombineParams.relation_json = os.path.join(output_base, 'relations')
c.CombineParams.salience_data = os.path.join(output_base, 'salience')
c.CombineParams.source_folder = os.path.join(output_base, 'txt')
c.CombineParams.add_rule_detector = True

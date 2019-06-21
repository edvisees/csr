import os

language = 'english'

c.AidaConvertParams.aida_annotation = \
    "/data/OPERA/LDC2019E07_AIDA_Phase_1_Evaluation_Practice_Topic_" \
    "Annotations_V6.0/data/R107/"
c.AidaConvertParams.ontology_path = \
    "../csr_resources/LDCOntology_v0.1.jsonld"
c.AidaConvertParams.parent_children_tab = \
    "/home/hector/workspace/aida/dev/LDC2019E04/parent_children.tab"
c.AidaConvertParams.source_folder = os.path.join(
    "/home/hector/workspace/aida/dev/LDC2019E04/ltf_output/hector_out/",
    language,
    "txt"
)
c.AidaConvertParams.brat_output_path = os.path.join(
    "/home/hector/public_html/brat_jun/data/aida/LDC2019E07_for_LDC2019E04",
    language,
    "R107"
)
c.AidaConvertParams.language = "en"

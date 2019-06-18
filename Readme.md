CSR (Central Semantic Representation)
===

The CSR format is a format for storing text analysis results, which is mainly
 developed for the AIDA (Active Interpretation of Disparate Alternatives) 
project.

This repository contains several toolkit for CSR analysis.

1. [csr](https://github.com/edvisees/csr/blob/master/event/io/csr.py) defines
the main format.
2. [csr2stuff](https://github.com/edvisees/csr/blob/master/event/io/csr2stuff.py)
allows to convert CSR format to other formats.
    - Brat annotation format (This is the format used by the 
    Brat (http://brat.nlplab.org) Annotation Tool). It allows one to view the
    annotation in context.
        ```
        python -m event.io.csr2stuff [csr_directory] [output_directory] 
          [ontology in json_ld format]
        ```
    - The ontology can be obtained [here](
    https://github.com/edvisees/csr_resources/blob/master/LDCOntology_v0.1.jsonld)
    
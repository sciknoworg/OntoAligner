# -*- coding: utf-8 -*-
__version__ = "1.3.0"

from .pipeline import OntoAlignerPipeline
from ontoaligner import ontology, base, encoder, ontology_matchers, utils, postprocess

__all__ = [
    "ontology",
    "base",
    "encoder",
    "ontology_matchers",
    "utils",
    "postprocess",
    "OntoAlignerPipeline"
]

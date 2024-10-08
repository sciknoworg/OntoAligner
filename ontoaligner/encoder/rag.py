# -*- coding: utf-8 -*-
from typing import Any

from .encoders import RAGEncoder
from .lightweight import ConceptLightweightEncoder


class ConceptRAGEncoder(RAGEncoder):
    items_in_owl: str = "(Concept)"
    retrieval_encoder: Any = ConceptLightweightEncoder
    llm_encoder: str = "LabelRAGDataset"


class ConceptChildrenRAGEncoder(RAGEncoder):
    items_in_owl: str = "(Concept, Children)"
    retrieval_encoder: Any = ConceptLightweightEncoder
    llm_encoder: str = "LabelChildrenRAGDataset"


class ConceptParentRAGEncoder(RAGEncoder):
    items_in_owl: str = "(Concept, Parent)"
    retrieval_encoder: Any = ConceptLightweightEncoder
    llm_encoder: str = "LabelParentRAGDataset"

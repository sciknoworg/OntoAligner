# -*- coding: utf-8 -*-
from .rag import ConceptRAGEncoder, ConceptParentRAGEncoder, ConceptChildrenRAGEncoder


class ConceptFewShotEncoder(ConceptRAGEncoder):
    llm_encoder: str = "ConceptFewShotDataset"


class ConceptChildrenFewShotEncoder(ConceptChildrenRAGEncoder):
    llm_encoder: str = "ConceptChildrenFewShotDataset"


class ConceptParentFewShotEncoder(ConceptParentRAGEncoder):
    llm_encoder: str = "ConceptParentFewShotDataset"

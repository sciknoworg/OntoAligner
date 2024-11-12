# -*- coding: utf-8 -*-
"""
This script defines three encoder classes for few-shot learning based on the RAG (retrieval-augmented generation) method.
These classes extend the functionality of the RAG-based encoders for concept, concept children, and concept parent,
specializing them for few-shot datasets related to concepts and their hierarchical relationships.

Classes:
    - ConceptFewShotEncoder: A few-shot learning encoder for concepts.
    - ConceptChildrenFewShotEncoder: A few-shot learning encoder for concept children.
    - ConceptParentFewShotEncoder: A few-shot learning encoder for concept parents.
"""
from .rag import ConceptRAGEncoder, ConceptParentRAGEncoder, ConceptChildrenRAGEncoder

class ConceptFewShotEncoder(ConceptRAGEncoder):
    """
    A few-shot learning encoder for concepts using retrieval-augmented generation (RAG).

    This class extends the ConceptRAGEncoder and is designed specifically for few-shot learning tasks
    related to concepts. It uses a custom few-shot dataset for encoding concepts.

    Attributes:
        llm_encoder (str): The dataset used for few-shot learning. In this case, it uses "ConceptFewShotDataset".
    """
    llm_encoder: str = "ConceptFewShotDataset"


class ConceptChildrenFewShotEncoder(ConceptChildrenRAGEncoder):
    """
    A few-shot learning encoder for concept children using retrieval-augmented generation (RAG).

    This class extends the ConceptChildrenRAGEncoder and is designed for few-shot learning tasks
    related to concept children. It uses a custom few-shot dataset for encoding concept children.

    Attributes:
        llm_encoder (str): The dataset used for few-shot learning. In this case, it uses "ConceptChildrenFewShotDataset".
    """
    llm_encoder: str = "ConceptChildrenFewShotDataset"


class ConceptParentFewShotEncoder(ConceptParentRAGEncoder):
    """
    A few-shot learning encoder for concept parents using retrieval-augmented generation (RAG).

    This class extends the ConceptParentRAGEncoder and is designed for few-shot learning tasks
    related to concept parents. It uses a custom few-shot dataset for encoding concept parents.

    Attributes:
        llm_encoder (str): The dataset used for few-shot learning. In this case, it uses "ConceptParentFewShotDataset".
    """
    llm_encoder: str = "ConceptParentFewShotDataset"

# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script defines three encoder classes that extend the `RAGEncoder` class to specialize in encoding OWL items
representing different ontology concepts. These encoders use a retrieval-based approach along with a language model
encoder for efficient handling of ontology mapping tasks.

Classes:
    - ConceptRAGEncoder: Encodes OWL items representing a Concept, with a retrieval encoder and a language model encoder.
    - ConceptChildrenRAGEncoder: Encodes OWL items representing a Concept and its Children, with a retrieval encoder and a language model encoder.
    - ConceptParentRAGEncoder: Encodes OWL items representing a Concept and its Parent, with a retrieval encoder and a language model encoder.
"""
from typing import Any

from .encoders import RAGEncoder
from .lightweight import ConceptLightweightEncoder

class ConceptRAGEncoder(RAGEncoder):
    """
    Encodes OWL items representing a Concept using retrieval-based and language model encoders.

    This class extends the `RAGEncoder` class and is specialized in encoding OWL items that consist of a Concept.
    The retrieval encoder uses the `ConceptLightweightEncoder` class to retrieve OWL items, while the language model
    encoder is set to "LabelRAGDataset".

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept.
        retrieval_encoder (Any): The retrieval encoder used for fetching OWL items, set to `ConceptLightweightEncoder`.
        llm_encoder (str): The language model encoder used, set to "LabelRAGDataset".
    """
    items_in_owl: str = "(Concept)"
    retrieval_encoder: Any = ConceptLightweightEncoder
    llm_encoder: str = "ConceptRAGDataset"


class ConceptChildrenRAGEncoder(RAGEncoder):
    """
    Encodes OWL items representing a Concept and its Children using retrieval-based and language model encoders.

    This class extends the `RAGEncoder` class and is specialized in encoding OWL items that consist of a Concept
    and its Children. The retrieval encoder uses the `ConceptLightweightEncoder` class to fetch the necessary items,
    while the language model encoder is set to "LabelChildrenRAGDataset".

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept and its Children.
        retrieval_encoder (Any): The retrieval encoder used for fetching OWL items, set to `ConceptLightweightEncoder`.
        llm_encoder (str): The language model encoder used, set to "LabelChildrenRAGDataset".
    """
    items_in_owl: str = "(Concept, Children)"
    retrieval_encoder: Any = ConceptLightweightEncoder
    llm_encoder: str = "ConceptChildrenRAGDataset"


class ConceptParentRAGEncoder(RAGEncoder):
    """
    Encodes OWL items representing a Concept and its Parent using retrieval-based and language model encoders.

    This class extends the `RAGEncoder` class and is specialized in encoding OWL items that consist of a Concept
    and its Parent. The retrieval encoder uses the `ConceptLightweightEncoder` class to retrieve the necessary items,
    while the language model encoder is set to "LabelParentRAGDataset".

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept and its Parent.
        retrieval_encoder (Any): The retrieval encoder used for fetching OWL items, set to `ConceptLightweightEncoder`.
        llm_encoder (str): The language model encoder used, set to "LabelParentRAGDataset".
    """
    items_in_owl: str = "(Concept, Parent)"
    retrieval_encoder: Any = ConceptLightweightEncoder
    llm_encoder: str = "ConceptParentRAGDataset"

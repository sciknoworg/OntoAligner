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

from ..base import BaseEncoder
from .lightweight import ConceptLightweightEncoder


class RAGEncoder(BaseEncoder):
    """
    A retrieval-augmented generation (RAG) encoder for ontology mapping.

    This class leverages retrieval-augmented generation for encoding ontology data,
    allowing for both retrieval of relevant data and generation of encoded information.
    """
    retrieval_encoder: Any = None
    llm_encoder: str = None

    def parse(self, **kwargs) -> Any:
        """
        Processes the source and target ontologies into indices for retrieval and encoding.

        This method converts the source and target ontologies into mappings of IRI to index,
        preparing them for use in a retrieval-augmented generation model.

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            dict: A dictionary with the retrieval encoder, LLM encoder, task arguments,
                  and the source and target ontology index mappings.
        """
        # self.dataset_module = kwargs["dataset-module"]
        source_onto_iri2index = {
            source["iri"]: index for index, source in enumerate(kwargs["source"])
        }
        target_onto_iri2index = {
            target["iri"]: index for index, target in enumerate(kwargs["target"])
        }
        return {
            "retriever-encoder": self.retrieval_encoder,
            "llm-encoder": self.llm_encoder,
            "task-args": kwargs,
            "source-onto-iri2index": source_onto_iri2index,
            "target-onto-iri2index": target_onto_iri2index,
        }

    def __str__(self):
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the encoder's name as key and items_in_owl as value.
        """
        return {"RagEncoder": self.items_in_owl}

    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder and its usage.

        Returns:
            str: A description of the encoder's components.
        """
        return "PROMPT-TEMPLATE USES:" + self.llm_encoder + " ENCODER"


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

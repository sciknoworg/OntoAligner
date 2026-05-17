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
from typing import Any, Dict, List
from collections import Counter

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


class OLaLaEncoder(BaseEncoder):
    """
    An encoder for preparing OLaLa parser output.
    """

    items_in_owl: str = "(OLaLa TextExtractorSet)"

    def get_host_uri_by_sampling(self, items: list, sample_size: int = 50) -> str:
        """
        Extracts the most common host by sampling ontology items.

        Parameters:
            items (list): The parsed ontology items.
            sample_size (int): The number of items to sample.

        Returns:
            str: The most common host.
        """
        hosts = []
        for item in items:
            hosts.append(item.get("olala").get("host", ""))
            if len(hosts) >= sample_size:
                break
        if not hosts:
            return ""
        return Counter(hosts).most_common(1)[0][0]


    def parse(self, **kwargs) -> Any:
        """
        Parses source and target ontologies into OLaLa-ready inputs.

        Parameters:
            **kwargs: Contains the source and target ontologies.

        Returns:
            list: A list containing prepared source and target ontology items.
        """
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_host = self.get_host_uri_by_sampling(source_onto)
        target_host = self.get_host_uri_by_sampling(target_onto)
        source_items = [
            self.get_owl_items(owl=source, expected_host=source_host)
            for source in source_onto
        ]
        target_items = [
            self.get_owl_items(owl=target, expected_host=target_host)
            for target in target_onto
        ]
        return [source_items, target_items]

    def extract_high_precision_texts(self, owl: Dict,
                                     normalized_uri_fragment: str,
                                     is_uri_fragment_normalization_valid: bool) -> List[str]:
        """
        Extracts normalized texts for the high-precision matcher.

        Parameters:
            owl (Dict): A parsed ontology item.
            normalized_uri_fragment (str): The URI fragment.

        Returns:
            List[str]: The normalized high-precision texts.
        """

        values = set()
        from rdflib import RDFS
        for record in owl.get("olala", {}).get("direct_string_literals", []):
            if record.get("predicate") != str(RDFS.label):
                continue
            normalized = record.get("normalized_text", "")
            if normalized:
                values.add(normalized)
        if is_uri_fragment_normalization_valid:
            values.add(normalized_uri_fragment)
        return list(values)

    def is_resource_for_sbert(self, host, expected_host) -> bool:
        """
        Checks whether a resource should be kept for SBERT candidate generation.
        """
        if host == "":
            return True
        return expected_host == host

    def get_owl_items(self, owl: Dict, expected_host: str) -> Dict:
        """
        Extracts OLaLa-ready fields from a parsed ontology item.

        Parameters:
            owl (Dict): A parsed ontology item.
            expected_host (str): The expected ontology host.
        Returns:
            Dict: The prepared OLaLa ontology item.
        """
        olala = owl.get("olala", {})
        iri = owl["iri"]
        label = owl.get("label", "")
        uri_fragment = olala.get("uri_fragment", "")
        host = olala.get("host", "")
        normalized_label = olala.get("normalized_label", "")
        normalized_uri_fragment = olala.get("normalized_uri_fragment", "")
        raw_texts = olala.get("text_extractor_set", [])
        sbert_texts = olala.get("normalized_text_extractor_set", [])

        only_label = olala.get("label", "")
        hp_texts = self.extract_high_precision_texts(
            owl=owl,
            normalized_uri_fragment = normalized_uri_fragment,
            is_uri_fragment_normalization_valid = olala.get("is_uri_fragment_normalization_valid", False)
        )

        return {
            "iri": iri,
            "label": label,
            "text": " ".join(sbert_texts),
            "texts": sbert_texts,
            "raw_texts": raw_texts,
            "only_label": only_label,
            "hp_texts": hp_texts,
            "uri_fragment": uri_fragment,
            "normalized_label": normalized_label,
            "normalized_uri_fragment": normalized_uri_fragment,
            "host": host,
            "expected_host": expected_host,
            "keep_for_sbert": self.is_resource_for_sbert(host, expected_host),
        }


    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder.

        Returns:
            str: A description of the encoder.
        """
        return "INPUT CONSISTS OF OLALA TEXTEXTRACTORSET, ONLYLABEL, AND HP FIELDS"


    def __str__(self):
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the encoder name and items in OWL.
        """
        return f"OLaLaEncoder{self.items_in_owl}"

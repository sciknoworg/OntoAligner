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
This script defines three encoder classes that inherit from the LightweightEncoder class.
These encoders are used to process and transform OWL (Web Ontology Language) items into a format suitable for downstream tasks.
Each encoder is specialized for different types of OWL items: Concept, Concept with Children, and Concept with Parent.

Classes:
    - ConceptLightweightEncoder: Encodes OWL items representing concepts.
    - ConceptChildrenLightweightEncoder: Encodes OWL items representing concepts and their children.
    - ConceptParentLightweightEncoder: Encodes OWL items representing concepts and their parents.
"""
from typing import Any, Dict

from ..base import BaseEncoder

class LightweightEncoder(BaseEncoder):
    """
    A lightweight encoder for parsing ontology data and preprocessing it.

    This class provides methods for parsing ontological data, applying text preprocessing,
    and formatting the data into a structure suitable for further processing.
    """
    def parse(self, **kwargs) -> Any:
        """
        Parses the source and target ontologies, applying preprocessing.

        This method extracts ontology items (IRI and label) from the source and target ontologies,
        applies text preprocessing to the labels, and returns the encoded data.

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            list: A list containing two elements, the processed source and target ontologies.
        """
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_ontos = []
        for source in source_onto:
            encoded_source = self.get_owl_items(owl=source)
            encoded_source["text"] = self.preprocess(encoded_source["text"])
            source_ontos.append(encoded_source)
        target_ontos = []
        for target in target_onto:
            encoded_target = self.get_owl_items(owl=target)
            encoded_target["text"] = self.preprocess(encoded_target["text"])
            target_ontos.append(encoded_target)
        return [source_ontos, target_ontos]

    def __str__(self):
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the class name as key and items_in_owl as value.
        """
        return {"LightweightEncoder": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Abstract method for extracting ontology data.

        This method should be implemented by subclasses to extract specific ontology data
        (e.g., IRI and label) from the provided ontology item.

        Parameters:
            owl (Dict): A dictionary representing an ontology item.

        Returns:
            Any: The extracted ontology data.
        """
        pass

    def get_encoder_info(self):
        """
        Provides information about the encoder.

        Returns:
            str: A description of the encoder's function in the overall pipeline.
        """
        return "INPUT CONSIST OF COMBINED INFORMATION TO FUZZY STRING MATCHING"


class ConceptLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts. The `get_owl_items` method retrieves the IRI and label of the concept.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept.
    """
    items_in_owl: str = """(Concept)"""

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri' and 'label' keys.

        Returns:
            Dict: A dictionary containing the IRI and label of the concept.
        """
        return {"iri": owl["iri"], "text": owl["label"]}


class ConceptChildrenLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts and their children.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts and their children. The `get_owl_items` method retrieves the IRI, label of the concept, and the labels of its children.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept with Children.
    """
    items_in_owl: str = "(Concept, Children)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept, along with the labels of its children, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'childrens' keys where 'childrens' is a list of children with 'label' attributes.

        Returns:
            Dict: A dictionary containing the IRI, label of the concept, and the concatenated labels of its children.
        """
        childrens = ", ".join([children["label"] for children in owl["childrens"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(childrens)}


class ConceptParentLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts and their parents.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts and their parents. The `get_owl_items` method retrieves the IRI, label of the concept, and the labels of its parents.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept with Parent.
    """
    items_in_owl: str = "(Concept, Parent)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept, along with the labels of its parents, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'parents' keys where 'parents' is a list of parents with 'label' attributes.

        Returns:
            Dict: A dictionary containing the IRI, label of the concept, and the concatenated labels of its parents.
        """
        parents = ", ".join([parent["label"] for parent in owl["parents"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(parents)}

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
from typing import Any, Dict

from ..base import BaseEncoder


class LLMEncoder(BaseEncoder):
    """
    A naive encoder for ontology alignment.
    """
    def parse(self, **kwargs) -> Any:
        """
        Processes the source and target ontologies into a prompt for ontology alignment.

        This method formats the source and target ontologies into a string representation,
        filling in a pre-defined template that includes ontology items (IRI and label).

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            list: A list containing the formatted prompt string for ontology matching.
        """
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_ontos = []
        for source in source_onto:
            encoded_source = self.get_owl_items(owl=source)
            # encoded_source["concept"] = self.preprocess(encoded_source["text"])
            source_ontos.append(encoded_source)
        target_ontos = []
        for target in target_onto:
            encoded_target = self.get_owl_items(owl=target)
            # encoded_target["concept"] = self.preprocess(encoded_target["text"])
            target_ontos.append(encoded_target)
        return [source_ontos, target_ontos]

    def __str__(self):
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the template and items_in_owl values.
        """
        return {"LLMEncoder": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> str:
        """
        Abstract method to extract ontology data as a string.

        This method should be implemented by subclasses to extract specific ontology data
        (e.g., IRI and label) from the provided ontology item.

        Parameters:
            owl (Dict): A dictionary representing an ontology item.

        Returns:
            str: The extracted ontology data as a string.
        """
        pass

    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder and its prompt template.

        Returns:
            str: A description of the encoder's components.
        """
        return "INPUT CONSIST OF A DICTIONARY THAT CONSIST OF INFORMATION FOR THE GIVEN SOURCE-TARGET ONTOLOGIES."


class ConceptLLMEncoder(LLMEncoder):
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
        return {"iri": owl["iri"], "concept": owl["label"]}


class ConceptChildrenLLMEncoder(LLMEncoder):
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
        return {"iri": owl["iri"], "concept": owl["label"], "childrens": str(childrens)}


class ConceptParentLLMEncoder(LLMEncoder):
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
        return {"iri": owl["iri"], "concept": owl["label"], "parents": str(parents)}

class PropertyLLMEncoder(LLMEncoder):
    """
    Encodes OWL/RDF items that represent properties.

    This class inherits from the `LLMEncoder` class and is designed to encode OWL/RDF property items.
    The `get_owl_items` method retrieves the IRI, label, and definition of the property.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Property.
    """
    items_in_owl: str = "(Property)"

    def get_owl_items(self, prop: Dict) -> Any:
        """
        Extracts the IRI, label, and definition of a property from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL/RDF property item, expected to contain
                        'iri', 'label', and optionally 'definition' keys.

        Returns:
            Dict: A dictionary containing the IRI, label, definition, and combined text of the property.
        """        
        label = prop.get("label", "")

        combined_text = label

        return {
            "iri": prop["iri"],
            "label": label,
            "text": combined_text,
        }

class PropertyFullTextLLMEncoder(LLMEncoder):
    """
    Encodes OWL/RDF items that represent properties with domain, range, inverse property, and definition.

    This class inherits from the `LLMEncoder` class and is designed to encode OWL/RDF property items.
    The `get_owl_items` method retrieves the IRI, label, definition, domain, range, and inverse property information.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case,
                            a Property with Definition, Domain, Range, and Inverse.
    """
    items_in_owl: str = "(Property, Domain, Range, Inverse)"

    def get_owl_items(self, prop: Dict) -> Any:
        label = prop.get("label", "")

        domain_text = (
            " ".join(prop.get("domain_text", []))
            if len(prop.get("domain_text", [])) > 0
            else ""
        )

        range_text = (
            " ".join(prop.get("range_text", []))
            if len(prop.get("range_text", [])) > 0
            else ""
        )

        inverse_text = ""
        if prop.get("inverse_of"):
            inverse_text = (
                " ".join(prop.get("inverse_label", []))
                if len(prop.get("inverse_label", [])) > 0
                else ""
            )

        combined_text = label

        if domain_text:
            combined_text += "  " + domain_text

        if range_text:
            combined_text += "  " + range_text

        if inverse_text:
            combined_text += "  inverse: " + inverse_text

        return {
            "iri": prop["iri"],
            "label": label,
            "domain": domain_text,
            "range": range_text,
            "inverse": inverse_text,
            "text": combined_text,
        }
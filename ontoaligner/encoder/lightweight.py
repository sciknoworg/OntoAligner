# -*- coding: utf-8 -*-
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

from .encoders import LightweightEncoder

class ConceptLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts. The `get_owl_items` method retrieves the IRI and label of the concept.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept.

    Methods:
        get_owl_items(owl: Dict) -> Dict:
            Given an OWL item, returns a dictionary containing the IRI and label of the concept.
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

    Methods:
        get_owl_items(owl: Dict) -> Dict:
            Given an OWL item, returns a dictionary containing the IRI, label of the concept,
            and labels of its children.
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

    Methods:
        get_owl_items(owl: Dict) -> Dict:
            Given an OWL item, returns a dictionary containing the IRI, label of the concept,
            and labels of its parents.
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

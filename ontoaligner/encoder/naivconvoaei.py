# -*- coding: utf-8 -*-
"""
This script defines three encoder classes that inherit from the `NaiveConvOAEIEncoder` class.
These encoders are used to transform OWL (Web Ontology Language) items into a format suitable for ontology mapping tasks.
Each encoder is specialized for different types of OWL items: Concept, Concept with Children, and Concept with Parent.

Classes:
    - ConceptNaiveEncoder: Encodes OWL items representing IRI and Concept.
    - ConceptChildrenNaiveEncoder: Encodes OWL items representing IRI, Concept, and Children.
    - ConceptParentNaiveEncoder: Encodes OWL items representing IRI, Concept, and Parent.
"""
from typing import Dict

from .encoders import NaiveConvOAEIEncoder


class ConceptNaiveEncoder(NaiveConvOAEIEncoder):
    """
    Encodes OWL items that represent IRI and Concept.

    This class inherits from the `NaiveConvOAEIEncoder` class and is designed to encode OWL items
    that consist of an IRI and a concept. The `get_owl_items` method retrieves the IRI and label of the concept.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, an IRI and Concept.

    Methods:
        get_owl_items(owl: Dict) -> str:
            Given an OWL item, returns a formatted string containing the IRI and label of the concept.
    """
    items_in_owl: str = "(IRI, Concept)"

    def get_owl_items(self, owl: Dict) -> str:
        """
        Extracts the IRI and label of a concept from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri' and 'label' keys.

        Returns:
            str: A formatted string containing the IRI and label of the concept.
        """
        return f"({owl['iri']}, {owl['label']}), "


class ConceptChildrenNaiveEncoder(NaiveConvOAEIEncoder):
    """
    Encodes OWL items that represent IRI, Concept, and Children.

    This class inherits from the `NaiveConvOAEIEncoder` class and is designed to encode OWL items
    that consist of an IRI, a concept, and its children. The `get_owl_items` method retrieves the IRI,
    label of the concept, and the labels of its children.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, an IRI, Concept, and Children.

    Methods:
        get_owl_items(owl: Dict) -> str:
            Given an OWL item, returns a formatted string containing the IRI, label of the concept, and labels of its children.
    """
    items_in_owl: str = "(IRI, Concept, Children)"

    def get_owl_items(self, owl: Dict) -> str:
        """
        Extracts the IRI and label of a concept, along with the labels of its children, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'childrens' keys where 'childrens' is a list of children with 'label' attributes.

        Returns:
            str: A formatted string containing the IRI, label of the concept, and the labels of its children.
        """
        childrens = [children["label"] for children in owl["childrens"]]
        return f"({owl['iri']}, {owl['label']}, {str(childrens)}), "


class ConceptParentNaiveEncoder(NaiveConvOAEIEncoder):
    """
    Encodes OWL items that represent IRI, Concept, and Parent.

    This class inherits from the `NaiveConvOAEIEncoder` class and is designed to encode OWL items
    that consist of an IRI, a concept, and its parents. The `get_owl_items` method retrieves the IRI,
    label of the concept, and the labels of its parents.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, an IRI, Concept, and Parent.

    Methods:
        get_owl_items(owl: Dict) -> str:
            Given an OWL item, returns a formatted string containing the IRI, label of the concept, and labels of its parents.
    """
    items_in_owl: str = "(IRI, Concept, Parent)"

    def get_owl_items(self, owl: Dict) -> str:
        """
        Extracts the IRI and label of a concept, along with the labels of its parents, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'parents' keys where 'parents' is a list of parents with 'label' attributes.

        Returns:
            str: A formatted string containing the IRI, label of the concept, and the labels of its parents.
        """
        parents = [parent["label"] for parent in owl["parents"]]
        return f"({owl['iri']}, {owl['label']}, {str(parents)}), "

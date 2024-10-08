# -*- coding: utf-8 -*-
from typing import Dict

from .encoders import NaiveConvOAEIEncoder


class ConceptNaiveEncoder(NaiveConvOAEIEncoder):
    items_in_owl: str = "(IRI, Concept)"

    def get_owl_items(self, owl: Dict) -> str:
        return f"({owl['iri']}, {owl['label']}), "


class ConceptChildrenNaiveEncoder(NaiveConvOAEIEncoder):
    items_in_owl: str = "(IRI, Concept, Children)"

    def get_owl_items(self, owl: Dict) -> str:
        childrens = [children["label"] for children in owl["childrens"]]
        return f"({owl['iri']}, {owl['label']}, {str(childrens)}), "


class ConceptParentNaiveEncoder(NaiveConvOAEIEncoder):
    items_in_owl: str = "(IRI, Concept, Parent)"

    def get_owl_items(self, owl: Dict) -> str:
        parents = [parent["label"] for parent in owl["parents"]]
        return f"({owl['iri']}, {owl['label']}, {str(parents)}), "

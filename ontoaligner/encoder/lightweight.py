# -*- coding: utf-8 -*-
from typing import Any, Dict

from .encoders import LightweightEncoder


class ConceptLightweightEncoder(LightweightEncoder):
    items_in_owl: str = """(Concept)"""

    def get_owl_items(self, owl: Dict) -> Any:
        return {"iri": owl["iri"], "text": owl["label"]}


class ConceptChildrenLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Concept, Children)"

    def get_owl_items(self, owl: Dict) -> Any:
        childrens = ", ".join([children["label"] for children in owl["childrens"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(childrens)}


class ConceptParentLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Concept, Parent)"

    def get_owl_items(self, owl: Dict) -> Any:
        parents = ", ".join([parent["label"] for parent in owl["parents"]])
        return {"iri": owl["iri"], "text": owl["label"] + "  " + str(parents)}

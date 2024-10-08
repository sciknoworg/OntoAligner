# -*- coding: utf-8 -*-
import os
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "food"


class FoodOntology(BaseOntologyParser):
    def is_contain_label(self, owl_class: Any) -> bool:
        if len(owl_class.prefLabel.en) == 0:
            return False
        return True

    def get_label(self, owl_class: Any) -> str:
        return str(owl_class.prefLabel.en.first())

    def get_synonyms(self, owl_class: Any) -> List:
        return []

    def get_parents(self, owl_class: Any) -> List:
        return []

    def get_comments(self, owl_class: Any) -> List:
        return []


class CiqualSirenOMDataset(OMDataset):
    track = track
    ontology_name = "ciqual-siren"

    source_ontology = FoodOntology()
    target_ontology = FoodOntology()

    working_dir = os.path.join(track, ontology_name)

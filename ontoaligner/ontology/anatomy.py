# -*- coding: utf-8 -*-
import os.path
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "anatomy"


class MouseOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.comment)


class HumanOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.hasDefinition)


class MouseHumanOMDataset(OMDataset):
    track = track
    ontology_name = "mouse-human"

    source_ontology = MouseOntology()
    target_ontology = HumanOntology()

    working_dir = os.path.join(track, ontology_name)

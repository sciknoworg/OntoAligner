# -*- coding: utf-8 -*-
import os.path
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "phenotype"


class DoidOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.hasExactSynonym


class OrdoOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.definition

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_synonyms(self, owl_class: Any) -> List:
        return []


class DoidOrdoOMDataset(OMDataset):
    track = track
    ontology_name = "doid-ordo"
    source_ontology = DoidOntology()
    target_ontology = OrdoOntology()
    working_dir = os.path.join(track, ontology_name)


class HpOntology(DoidOntology):
    def is_contain_label(self, owl_class: Any) -> bool:
        try:
            if len(owl_class.label) == 0:
                return False
            if "/HP_" in owl_class.iri:
                return True
        except Exception as e:
            print(f"Exception: {e}")
        return False


class MpOntology(DoidOntology):
    def is_contain_label(self, owl_class: Any) -> bool:
        try:
            if len(owl_class.label) == 0:
                return False
            if "/MP_" in owl_class.iri:
                return True
        except Exception as e:
            print(f"Exception: {e}")
        return False


class HpMpOMDataset(OMDataset):
    track = track
    ontology_name = "hp-mp"
    source_ontology = HpOntology()
    target_ontology = MpOntology()
    working_dir = os.path.join(track, ontology_name)

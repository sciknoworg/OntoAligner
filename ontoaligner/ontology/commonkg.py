# -*- coding: utf-8 -*-
import os.path
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "commonkg"


class CommonKGOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return []

    def get_synonyms(self, owl_class: Any) -> List:
        return []


class NellDbpediaOMDataset(OMDataset):
    track = track
    ontology_name = "nell-dbpedia"
    source_ontology = CommonKGOntology()
    target_ontology = CommonKGOntology()
    working_dir = os.path.join(track, ontology_name)


class YagoWikidataOMDataset(OMDataset):
    track = track
    ontology_name = "yago-wikidata"
    source_ontology = CommonKGOntology()
    target_ontology = CommonKGOntology()
    working_dir = os.path.join(track, ontology_name)

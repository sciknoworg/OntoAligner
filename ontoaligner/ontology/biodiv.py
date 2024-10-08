# -*- coding: utf-8 -*-
import os.path
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "biodiv"


class EnvoOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment

    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.hasRelatedSynonym


class SweetOntology(BaseOntologyParser):
    def is_contain_label(self, owl_class: Any) -> bool:
        try:
            if owl_class.name == "Thing":
                return False
            if len(owl_class.prefixIRI) == 0:
                return False
            return True
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def get_label(self, owl_class: Any) -> str:
        return str(owl_class.prefixIRI.first())

    def get_comments(self, owl_class: Any) -> List:
        return []

    def get_synonyms(self, owl_class: Any) -> List:
        return []


class EnvoSweetOMDataset(OMDataset):
    track = track
    ontology_name = "envo-sweet"
    source_ontology = EnvoOntology()
    target_ontology = SweetOntology()
    working_dir = os.path.join(track, ontology_name)


class SeaLifeOntology(BaseOntologyParser):
    def is_contain_label(self, owl_class: Any) -> bool:
        if len(owl_class.label.en) == 0:
            return False
        return True

    def get_label(self, owl_class: Any) -> str:
        return str(owl_class.label.en.first())

    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.hasRelatedSynonym.en

    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment.en


class FishZooplanktonOMDataset(OMDataset):
    track = track
    ontology_name = "fish-zooplankton"
    source_ontology = SeaLifeOntology()
    target_ontology = SeaLifeOntology()
    working_dir = os.path.join(track, ontology_name)


class MacroalgaeMacrozoobenthosOMDataset(OMDataset):
    track = track
    ontology_name = "macroalgae-macrozoobenthos"
    source_ontology = SeaLifeOntology()
    target_ontology = SeaLifeOntology()
    working_dir = os.path.join(track, ontology_name)


class TAXREFLDOntology(BaseOntologyParser):
    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.hasSynonym

    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment.en


class NCBIOntology(BaseOntologyParser):
    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.hasRelatedSynonym

    def get_comments(self, owl_class: Any) -> List:
        return []


class TaxrefldBacteriaNcbitaxonBacteriaOMDataset(OMDataset):
    track = track
    ontology_name = "taxrefldBacteria-ncbitaxonBacteria"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldChromistaNcbitaxonChromistaOMDataset(OMDataset):
    track = track
    ontology_name = "taxrefldChromista-ncbitaxonChromista"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldFungiNcbitaxonFungiOMDataset(OMDataset):
    track = track
    ontology_name = "taxrefldFungi-ncbitaxonFungi"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldPlantaeNcbitaxonPlantaeOMDataset(OMDataset):
    track = track
    ontology_name = "taxrefldPlantae-ncbitaxonPlantae"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldProtozoaNcbitaxonProtozoaOMDataset(OMDataset):
    track = track
    ontology_name = "taxrefldProtozoa-ncbitaxonProtozoa"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)

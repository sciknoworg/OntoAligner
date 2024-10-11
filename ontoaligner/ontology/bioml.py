# -*- coding: utf-8 -*-
import os.path
from typing import Any, Dict, List

from ..base import BaseAlignmentsParser, BaseOntologyParser, OMDataset
from ..utils import io

track = "bio-ml"


def refactor_tsv(dataframe: Any, columns: Dict) -> List:
    rows, keys_names = [], []
    for source_column, column_target_name in columns.items():
        rows.append(dataframe[source_column].tolist())
        keys_names.append(column_target_name)
    refactored_tsv = []
    for items in list(zip(*rows)):
        item_dict = {}
        for index in range(len(items)):
            if keys_names[index] == "candidates":
                item_dict[keys_names[index]] = list(eval(items[index]))
            else:
                item_dict[keys_names[index]] = items[index]
        refactored_tsv.append(item_dict)
    return refactored_tsv


class BioMLAlignmentsParser(BaseAlignmentsParser):
    def parse(self, input_file_path: str = None) -> Dict:
        references = {
            "test-cands": refactor_tsv(
                dataframe=io.read_tsv(input_file_path),
                columns={
                    "SrcEntity": "source",
                    "TgtEntity": "target",
                    "TgtCandidates": "candidates",
                },
            )
        }
        return references

class BioOntology(BaseOntologyParser):
    def __init__(self):
        super().__init__()

    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment

    def get_synonyms(self, owl_class: Any) -> List:
        try:
            syn = owl_class.hasRelatedSynonym + owl_class.hasExactSynonym
            return list(set(syn))
        except Exception:
            pass
        try:
            return owl_class.hasRelatedSynonym
        except Exception:
            pass
        try:
            return owl_class.hasExactSynonym
        except Exception:
            return []

class NCITDOIDDiseaseOMDataset(OMDataset):
    track = track
    ontology_name = "ncit-doid.disease"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class OMIMORDODiseaseOMDataset(OMDataset):
    track = track
    ontology_name = "omim-ordo.disease"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDFMABodyOMDataset(OMDataset):
    track = track
    ontology_name = "snomed-fma.body"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDNCITNeoplasOMDataset(OMDataset):
    track = track
    ontology_name = "snomed-ncit.neoplas"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDNCITPharmOMDataset(OMDataset):
    track = track
    ontology_name = "snomed-ncit.pharm"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDFMABodyLLMOMDataset(OMDataset):
    track = "bio-llm"
    ontology_name = "snomed-fma.body"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class NCITDOIDDiseaseLLMOMDataset(OMDataset):
    track = "bio-llm"
    ontology_name = "ncit-doid.disease"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()

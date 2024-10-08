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
    def parse(self, root_dir: str, reference_file_name: str = None) -> Dict:
        print(f"\t\tworking on reference: {root_dir}")
        references = {
            "equiv": {
                "full": refactor_tsv(
                    dataframe=io.read_tsv(
                        os.path.join(root_dir, "refs_equiv", "full.tsv")
                    ),
                    columns={"SrcEntity": "source", "TgtEntity": "target"},
                ),
                "test": refactor_tsv(
                    dataframe=io.read_tsv(
                        os.path.join(root_dir, "refs_equiv", "test.tsv")
                    ),
                    columns={"SrcEntity": "source", "TgtEntity": "target"},
                ),
                "test-cands": refactor_tsv(
                    dataframe=io.read_tsv(
                        os.path.join(root_dir, "refs_equiv", "test.cands.tsv")
                    ),
                    columns={
                        "SrcEntity": "source",
                        "TgtEntity": "target",
                        "TgtCandidates": "candidates",
                    },
                ),
                "train": refactor_tsv(
                    dataframe=io.read_tsv(
                        os.path.join(root_dir, "refs_equiv", "train.tsv")
                    ),
                    columns={"SrcEntity": "source", "TgtEntity": "target"},
                ),
            },
            "subs": {
                "test-cands": refactor_tsv(
                    dataframe=io.read_tsv(
                        os.path.join(root_dir, "refs_subs", "test.cands.tsv")
                    ),
                    columns={
                        "SrcEntity": "source",
                        "TgtEntity": "target",
                        "TgtCandidates": "candidates",
                    },
                ),
                "train": refactor_tsv(
                    dataframe=io.read_tsv(
                        os.path.join(root_dir, "refs_subs", "train.tsv")
                    ),
                    columns={"SrcEntity": "source", "TgtEntity": "target"},
                ),
            },
        }
        return references


class BioLLMAlignmentsParser(BaseAlignmentsParser):
    def parse(self, root_dir: str, reference_file_name: str = None) -> Dict:
        print(f"\t\tworking on reference: {root_dir}")
        references = {
            "test-cands": refactor_tsv(
                dataframe=io.read_tsv(os.path.join(root_dir, "test_cands.tsv")),
                columns={
                    "SrcEntity": "source",
                    "TgtEntity": "target",
                    "TgtCandidates": "candidates",
                },
            )
        }
        return references


class BioMLOMDataset(OMDataset):
    def collect(self, root_dir: str) -> Dict:
        om_root_path = os.path.join(root_dir, self.track, self.ontology_name)
        data = {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source": self.source_ontology.parse(root_dir=om_root_path),
            "target": self.target_ontology.parse(root_dir=om_root_path),
            "reference": self.alignments.parse(root_dir=om_root_path),
        }
        return data


class BioOntology(BaseOntologyParser):
    def __init__(self, ontology_file_name):
        super().__init__()
        self.ontology_file_name = ontology_file_name

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

    def parse(self, root_dir: str, ontology_file_name: str = None) -> List:
        return super().parse(
            root_dir=root_dir, ontology_file_name=self.ontology_file_name
        )


class NCITDOIDDiseaseOMDataset(BioMLOMDataset):
    track = track
    ontology_name = "ncit-doid.disease"
    source_ontology = BioOntology(ontology_file_name="ncit.owl")
    target_ontology = BioOntology(ontology_file_name="doid.owl")
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class OMIMORDODiseaseOMDataset(BioMLOMDataset):
    track = track
    ontology_name = "omim-ordo.disease"
    source_ontology = BioOntology(ontology_file_name="omim.owl")
    target_ontology = BioOntology(ontology_file_name="ordo.owl")
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDFMABodyOMDataset(BioMLOMDataset):
    track = track
    ontology_name = "snomed-fma.body"
    source_ontology = BioOntology(ontology_file_name="snomed.body.owl")
    target_ontology = BioOntology(ontology_file_name="fma.body.owl")
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDNCITNeoplasOMDataset(BioMLOMDataset):
    track = track
    ontology_name = "snomed-ncit.neoplas"
    source_ontology = BioOntology(ontology_file_name="snomed.neoplas.owl")
    target_ontology = BioOntology(ontology_file_name="ncit.neoplas.owl")
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDNCITPharmOMDataset(BioMLOMDataset):
    track = track
    ontology_name = "snomed-ncit.pharm"
    source_ontology = BioOntology(ontology_file_name="snomed.pharm.owl")
    target_ontology = BioOntology(ontology_file_name="ncit.pharm.owl")
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDFMABodyLLMOMDataset(BioMLOMDataset):
    track = "bio-llm"
    ontology_name = "snomed-fma.body"
    source_ontology = BioOntology(ontology_file_name="snomed.body.owl")
    target_ontology = BioOntology(ontology_file_name="fma.body.owl")
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioLLMAlignmentsParser()


class NCITDOIDDiseaseLLMOMDataset(BioMLOMDataset):
    track = "bio-llm"
    ontology_name = "ncit-doid.disease"
    source_ontology = BioOntology(ontology_file_name="ncit.owl")
    target_ontology = BioOntology(ontology_file_name="doid.owl")
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioLLMAlignmentsParser()

# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script defines ontology parsers and datasets for bioinformatics-related tasks, specifically for
processing disease-related ontologies and alignment data. The main objective is to handle ontological data, parse TSV
files for specific bio-entities, and provide aligned datasets for various ontology combinations.
"""

import os.path
import pandas as pd

from typing import Any, Dict, List

from ...base import BaseAlignmentsParser, BaseOntologyParser, OMDataset

track = "bio-ml"


def refactor_tsv(dataframe: Any, columns: Dict) -> List:
    """
    Refactors a dataframe to a list of dictionaries with specified column mappings.

    This function restructures a DataFrame into a list of dictionaries, where each
    dictionary represents a row and the columns are renamed as per the specified mapping.

    Parameters:
        dataframe (Any): The input dataframe to be refactored.
        columns (Dict): A dictionary mapping the source column names to the target column names.

    Returns:
        List: A list of dictionaries where each dictionary represents a row in the TSV with
              the new column names and values.
    """
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
    """
    A class for parsing and processing bio-alignment data in TSV format.

    Inherits from `BaseAlignmentsParser`. This class handles the parsing of alignment
    data where each row contains source and target entity information, and the
    candidates for the target entity.
    """
    def parse(self, input_file_path: str = None) -> Dict:
        """
        Parses the input TSV file to extract alignment data.

        This method reads a TSV file, refactors the data, and organizes it into
        a dictionary format where the keys are "test-cands" and the values are
        the source-target alignment pairs.

        Parameters:
            input_file_path (str): The path to the TSV file to be parsed.

        Returns:
            Dict: A dictionary containing the "test-cands" data as key and a list of
                  aligned source and target entities as values.
        """
        references = {
            "test-cands": refactor_tsv(
                dataframe=pd.read_csv(input_file_path, sep="\t"),
                columns={
                    "SrcEntity": "source",
                    "TgtEntity": "target",
                    "TgtCandidates": "candidates",
                },
            )
        }
        return references


class BioOntology(BaseOntologyParser):
    """
    A class for parsing biological ontologies to extract comments and synonyms.

    Inherits from `BaseOntologyParser`. This class specifically extracts comments
    and synonyms (both related and exact) from biological ontology classes.
    """
    def __init__(self):
        """
        Initializes the BioOntology parser by calling the superclass constructor.
        """
        super().__init__()

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves the comments associated with the given ontology class.

        Parameters:
            owl_class (Any): The ontology class from which to extract comments.

        Returns:
            List: A list of comments associated with the ontology class.
        """
        return owl_class.comment

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms associated with the given ontology class.

        This method first tries to return both related and exact synonyms. If neither
        are found, it returns related synonyms or exact synonyms individually, depending
        on what is available.

        Parameters:
            owl_class (Any): The ontology class from which to extract synonyms.

        Returns:
            List: A list of synonyms associated with the ontology class.
        """
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
    """
    A dataset class combining the NCIT (National Cancer Institute Thesaurus) and DOID (Disease Ontology)
    ontologies for disease-related data.

    This class provides the source and target ontologies (both `BioOntology`), along with the
    working directory and alignment parser.
    """
    track = track
    ontology_name = "ncit-doid.disease"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class OMIMORDODiseaseOMDataset(OMDataset):
    """
    A dataset class combining the OMIM (Online Mendelian Inheritance in Man) and ORDO (Orphanet Rare Disease Ontology)
    ontologies for disease-related data.
    """
    track = track
    ontology_name = "omim-ordo.disease"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDFMABodyOMDataset(OMDataset):
    """
    A dataset class combining the SNOMED CT (Systematized Nomenclature of Medicine) and FMA (Foundational Model of Anatomy)
    ontologies for body-related data.

    This dataset is tailored for body-related ontological data, using SNOMED and FMA ontologies as sources and targets.
    """
    track = track
    ontology_name = "snomed-fma.body"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDNCITNeoplasOMDataset(OMDataset):
    """
    A dataset class combining the SNOMED CT (Systematized Nomenclature of Medicine) and NCIT (National Cancer Institute Thesaurus)
    ontologies for neoplasms (tumors) data.

    This dataset is used for neoplasms-related data with SNOMED and NCIT ontologies as source and target.
    """
    track = track
    ontology_name = "snomed-ncit.neoplas"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDNCITPharmOMDataset(OMDataset):
    """
    A dataset class combining the SNOMED CT (Systematized Nomenclature of Medicine) and NCIT (National Cancer Institute Thesaurus)
    ontologies for pharmaceutical data.

    This dataset uses SNOMED and NCIT ontologies for pharmacological information, suitable for drug-related data.
    """
    track = track
    ontology_name = "snomed-ncit.pharm"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class SNOMEDFMABodyLLMOMDataset(OMDataset):
    """
    A dataset class combining the SNOMED CT (Systematized Nomenclature of Medicine) and FMA (Foundational Model of Anatomy)
    ontologies for body-related data, specifically for Large Language Models (LLM).

    This dataset is used for LLM applications, combining SNOMED and FMA ontologies for body-related data.
    """
    track = "bio-llm"
    ontology_name = "snomed-fma.body"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()


class NCITDOIDDiseaseLLMOMDataset(OMDataset):
    """
    A dataset class combining the NCIT (National Cancer Institute Thesaurus) and DOID (Disease Ontology)
    ontologies for disease-related data, specifically for Large Language Models (LLM).

    This dataset is designed for LLM applications, combining NCIT and DOID ontologies for disease-related data.
    """
    track = "bio-llm"
    ontology_name = "ncit-doid.disease"
    source_ontology = BioOntology()
    target_ontology = BioOntology()
    working_dir = os.path.join(track, ontology_name)
    alignments: BaseAlignmentsParser = BioMLAlignmentsParser()

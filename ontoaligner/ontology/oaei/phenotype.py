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
This script defines ontology parsers for various disease-related ontologies, including DOID (Disease Ontology), ORDO (Orphanet Rare Disease Ontology),
HP (Human Phenotype Ontology), and MP (Mammalian Phenotype Ontology). It also defines dataset classes that map between source and target ontologies
for different phenotype-related datasets.
"""

import os.path
from typing import Any, List

from ...base import BaseOntologyParser, OMDataset

track = "phenotype"


class DoidOntology(BaseOntologyParser):
    """
    A parser for the DOID (Disease Ontology) ontology.

    This class extracts comments, labels, and synonyms for ontology classes.
    """

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves the comments for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose comments are to be retrieved.

        Returns:
            List: A list of comments for the given ontology class.
        """
        return owl_class.comment

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves the label for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose label is to be retrieved.

        Returns:
            str: The label of the ontology class.
        """
        return owl_class.label.first()

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose synonyms are to be retrieved.

        Returns:
            List: A list of synonyms for the given ontology class.
        """
        return owl_class.hasExactSynonym


class OrdoOntology(BaseOntologyParser):
    """
    A parser for the ORDO (Orphanet Rare Disease Ontology).

    This class extracts comments, labels, and synonyms for ontology classes.
    """

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves the comments for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose comments are to be retrieved.

        Returns:
            List: A list of comments for the given ontology class.
        """
        return owl_class.definition

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves the label for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose label is to be retrieved.

        Returns:
            str: The label of the ontology class.
        """
        return owl_class.label.first()

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose synonyms are to be retrieved.

        Returns:
            List: An empty list as no synonyms are implemented for this ontology class.
        """
        return []


class DoidOrdoOMDataset(OMDataset):
    """
    A dataset class for mapping between the DOID and ORDO ontologies.

    This class configures the source ontology as `DoidOntology` and the target ontology as
    `OrdoOntology` for the `doid-ordo` dataset. It specifies the working directory for the dataset.
    """
    track = track
    ontology_name = "doid-ordo"
    source_ontology = DoidOntology()
    target_ontology = OrdoOntology()
    working_dir = os.path.join(track, ontology_name)


class HpOntology(DoidOntology):
    """
    A parser for the HP (Human Phenotype Ontology) ontology.

    This class extends `DoidOntology` and provides additional functionality to check if
    the ontology class is related to HP by inspecting its IRI.
    """

    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if the ontology class contains a label and is related to HP.

        Parameters:
            owl_class (Any): The ontology class to check.

        Returns:
            bool: True if the class has a label and is related to HP, otherwise False.
        """
        try:
            if len(owl_class.label) == 0:
                return False
            if "/HP_" in owl_class.iri:
                return True
        except Exception as e:
            print(f"Exception: {e}")
        return False


class MpOntology(DoidOntology):
    """
    A parser for the MP (Mammalian Phenotype Ontology).

    This class extends `DoidOntology` and provides additional functionality to check if
    the ontology class is related to MP by inspecting its IRI.
    """

    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if the ontology class contains a label and is related to MP.

        Parameters:
            owl_class (Any): The ontology class to check.

        Returns:
            bool: True if the class has a label and is related to MP, otherwise False.
        """
        try:
            if len(owl_class.label) == 0:
                return False
            if "/MP_" in owl_class.iri:
                return True
        except Exception as e:
            print(f"Exception: {e}")
        return False


class HpMpOMDataset(OMDataset):
    """
    A dataset class for mapping between the HP and MP ontologies.

    This class configures the source ontology as `HpOntology` and the target ontology as
    `MpOntology` for the `hp-mp` dataset. It specifies the working directory for the dataset.
    """
    track = track
    ontology_name = "hp-mp"
    source_ontology = HpOntology()
    target_ontology = MpOntology()
    working_dir = os.path.join(track, ontology_name)

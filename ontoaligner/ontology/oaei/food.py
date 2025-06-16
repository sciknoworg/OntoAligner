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
This script defines classes for parsing and processing datasets related to food ontologies.
It includes a class for parsing the `FoodOntology` and a dataset configurations.
"""

import os
from typing import Any, List

from ...base import BaseOntologyParser, OMDataset

track = "food"


class FoodOntology(BaseOntologyParser):
    """
    A class for parsing and handling the Food ontology.

    This class provides methods for extracting information such as labels, synonyms,
    parent classes, and comments for ontology classes.
    """
    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if the ontology class has a label.

        Parameters:
            owl_class (Any): The ontology class whose label presence is to be checked.

        Returns:
            bool: True if the ontology class contains a label, otherwise False.
        """
        if len(owl_class.prefLabel.en) == 0:
            return False
        return True

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves the label for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose label is to be retrieved.

        Returns:
            str: The label of the ontology class.
        """
        return str(owl_class.prefLabel.en.first())

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves the synonyms for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose synonyms are to be retrieved.

        Returns:
            List: An empty list, as no synonyms are implemented for the Food ontology.
        """
        return []

    def get_parents(self, owl_class: Any) -> List:
        """
        Retrieves the parent classes for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose parents are to be retrieved.

        Returns:
            List: An empty list, as no parent classes are implemented for the Food ontology.
        """
        return []

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves the comments for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose comments are to be retrieved.

        Returns:
            List: An empty list, as no comments are implemented for the Food ontology.
        """
        return []


class CiqualSirenOMDataset(OMDataset):
    """
    A dataset class combining the Ciqual-Siren ontology for food-related data.

    This class uses the `FoodOntology` for both the source and target ontologies
    and defines the working directory for the dataset.
    """
    track = track
    ontology_name = "ciqual-siren"

    source_ontology = FoodOntology()
    target_ontology = FoodOntology()

    working_dir = os.path.join(track, ontology_name)

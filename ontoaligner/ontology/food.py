# -*- coding: utf-8 -*-
"""
This script defines classes for parsing and processing datasets related to food ontologies.
It includes a class for parsing the `FoodOntology` and a dataset configuration for the
Ciqual-Siren dataset.

The script contains the following components:
- `FoodOntology`: A class for handling the Food ontology, with methods for retrieving
  labels, synonyms, parents, and comments for ontology classes.
- `CiqualSirenOMDataset`: A class representing the Ciqual-Siren dataset, which combines
  source and target ontologies with a working directory configuration.

Classes:
- `FoodOntology`: A parser for the Food ontology, providing methods for handling ontology classes.
- `CiqualSirenOMDataset`: A dataset class that uses the FoodOntology for both the source
  and target ontologies and defines the working directory for the Ciqual-Siren dataset.

"""

import os
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "food"


class FoodOntology(BaseOntologyParser):
    """
    A class for parsing and handling the Food ontology.

    This class provides methods for extracting information such as labels, synonyms,
    parent classes, and comments for ontology classes.

    Methods:
        is_contain_label(owl_class: Any) -> bool: Checks if the ontology class has a label.
        get_label(owl_class: Any) -> str: Retrieves the label for the given ontology class.
        get_synonyms(owl_class: Any) -> List: Retrieves the synonyms for the given ontology class.
        get_parents(owl_class: Any) -> List: Retrieves the parent classes for the given ontology class.
        get_comments(owl_class: Any) -> List: Retrieves the comments for the given ontology class.
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

    Attributes:
        track (str): The track associated with this dataset, set to "food".
        ontology_name (str): The name of the dataset, set to "ciqual-siren".
        source_ontology (FoodOntology): The source ontology parser for Ciqual-Siren.
        target_ontology (FoodOntology): The target ontology parser for Ciqual-Siren.
        working_dir (str): The directory where the dataset files are stored, based on the track and ontology name.
    """
    track = track
    ontology_name = "ciqual-siren"

    source_ontology = FoodOntology()
    target_ontology = FoodOntology()

    working_dir = os.path.join(track, ontology_name)

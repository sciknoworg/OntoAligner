# -*- coding: utf-8 -*-
"""
This script defines ontology parsers for various disease-related ontologies,
including DOID (Disease Ontology), ORDO (Orphanet Rare Disease Ontology),
HP (Human Phenotype Ontology), and MP (Mammalian Phenotype Ontology).
It also defines dataset classes that map between source and target ontologies
for different phenotype-related datasets.

Classes:
- DoidOntology: Parses the DOID ontology, extracting comments, labels, and synonyms.
- OrdoOntology: Parses the ORDO ontology, extracting comments, labels, and synonyms.
- HpOntology: Extends DoidOntology, adding functionality to check if a label is present for HP-related classes.
- MpOntology: Extends DoidOntology, adding functionality to check if a label is present for MP-related classes.
- DoidOrdoOMDataset: A dataset class for mapping between the DOID and ORDO ontologies.
- HpMpOMDataset: A dataset class for mapping between the HP and MP ontologies.

"""

import os.path
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "phenotype"


class DoidOntology(BaseOntologyParser):
    """
    A parser for the DOID (Disease Ontology) ontology.

    This class extracts comments, labels, and synonyms for ontology classes.

    Methods:
        get_comments(owl_class: Any) -> List: Retrieves comments for the given ontology class.
        get_label(owl_class: Any) -> str: Retrieves the label for the given ontology class.
        get_synonyms(owl_class: Any) -> List: Retrieves synonyms for the given ontology class.
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

    Methods:
        get_comments(owl_class: Any) -> List: Retrieves comments for the given ontology class.
        get_label(owl_class: Any) -> str: Retrieves the label for the given ontology class.
        get_synonyms(owl_class: Any) -> List: Retrieves synonyms for the given ontology class.
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

    Attributes:
        track (str): The dataset's track name (phenotype).
        ontology_name (str): The name of the ontology (`doid-ordo`).
        source_ontology (DoidOntology): The source ontology used for mapping.
        target_ontology (OrdoOntology): The target ontology used for mapping.
        working_dir (str): The working directory path for the dataset.
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

    Methods:
        is_contain_label(owl_class: Any) -> bool: Checks if the ontology class contains a label and is related to HP.
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

    Methods:
        is_contain_label(owl_class: Any) -> bool: Checks if the ontology class contains a label and is related to MP.
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

    Attributes:
        track (str): The dataset's track name (phenotype).
        ontology_name (str): The name of the ontology (`hp-mp`).
        source_ontology (HpOntology): The source ontology used for mapping.
        target_ontology (MpOntology): The target ontology used for mapping.
        working_dir (str): The working directory path for the dataset.
    """
    track = track
    ontology_name = "hp-mp"
    source_ontology = HpOntology()
    target_ontology = MpOntology()
    working_dir = os.path.join(track, ontology_name)

# -*- coding: utf-8 -*-
"""
This script defines several classes for parsing and processing ontologies related to
material science and engineering. It includes helper functions for string manipulation and
ontology parsing methods for extracting labels, comments, synonyms, ancestors,
and other ontology-related information.
"""

import re
from typing import Any, List

import ontospy

from ..base import BaseOntologyParser, OMDataset

track = "mse"


def split_string(input_str):
    """
    Splits an input string into meaningful components based on a predefined regular expression pattern.

    The function looks for patterns in the input string and splits it into uppercase and lowercase parts,
    returning them as individual components. If the pattern is not found, it falls back to splitting
    the string based on uppercase and lowercase transitions.

    Parameters:
        input_str (str): The string to be split and formatted.

    Returns:
        str: A formatted string with components separated by spaces.
    """
    # Define a regular expression pattern to capture the desired components
    pattern = r"([A-Z]+)(\d+)([A-Z][a-z]+)?([A-Z][a-z]+)?"
    # Use re.findall to find all matching patterns in the input string
    matches = re.findall(pattern, input_str)
    if matches:
        # The first element of each match is the whole match, so we need to slice from the second element
        result = [match[0:] for match in matches[0]]
        # Filter out empty strings from the result
        result = [component for component in result if component]
    else:
        result = re.findall("[A-Z][^A-Z]*", input_str)
    result = " ".join(result)
    return result


class EMMOOntology(BaseOntologyParser):
    """
    A parser for the EMMO (European Materials & Modelling Ontology).

    This class provides methods for extracting information such as labels, comments,
    ancestors, and checking whether an ontology class contains a label.
    """
    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if the ontology class has a label.

        Parameters:
            owl_class (Any): The ontology class whose label presence is to be checked.

        Returns:
            bool: True if the ontology class contains a label, otherwise False.
        """
        try:
            if str(owl_class) == "owl.Thing":
                return False
            if len(owl_class.prefLabel) == 0:
                return False
            return True
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves the comments for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose comments are to be retrieved.

        Returns:
            List: The comments associated with the ontology class.
        """
        return owl_class.comment.en

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves and formats the label for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose label is to be retrieved.

        Returns:
            str: The formatted label of the ontology class.
        """
        return split_string(owl_class.prefLabel.en.first())

    def get_ancestors(self, owl_class: Any) -> List:
        """
        Retrieves the ancestors for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose ancestors are to be retrieved.

        Returns:
            List: A list of ancestor classes for the given ontology class.
        """
        return self.get_owl_items(list(owl_class.ancestors()))

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves the synonyms for the given ontology class.

        Parameters:
            owl_class (Any): The ontology class whose synonyms are to be retrieved.

        Returns:
            List: An empty list as no synonyms are implemented for this ontology class.
        """
        return []


class MaterialInformationOntoOntology(BaseOntologyParser):
    """
    A parser for the Material Information Ontology.

    This class provides methods for handling ontology items such as labels, names, IRIs,
    parents, children, and more. It also provides functionality to load the ontology from a file.
    """
    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if the ontology class has a label.

        Parameters:
            owl_class (Any): The ontology class whose label presence is to be checked.

        Returns:
            bool: Always returns True as all classes are assumed to have labels.
        """
        return True

    def get_name(self, owl_class: Any) -> str:
        """
        Retrieves the name of the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose name is to be retrieved.

        Returns:
            str: The name of the ontology class.
        """
        return str(owl_class.uri).split("#")[1]

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves and formats the label of the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose label is to be retrieved.

        Returns:
            str: The formatted label of the ontology class.
        """
        preprocessed_str = (
            self.get_iri(owl_class).split("#")[1].replace("_", " ").replace("-", "")
        )
        return split_string(preprocessed_str)

    def get_iri(self, owl_class: Any) -> str:
        """
        Retrieves the IRI of the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose IRI is to be retrieved.

        Returns:
            str: The IRI of the ontology class.
        """
        return str(owl_class.uri)

    def get_childrens(self, owl_class: Any) -> List:
        """
        Retrieves the children of the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose children are to be retrieved.

        Returns:
            List: A list of child classes for the given ontology class.
        """
        return self.get_owl_items(owl_class.children())

    def get_parents(self, owl_class: Any) -> List:
        """
        Retrieves the parents of the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose parents are to be retrieved.

        Returns:
            List: A list of parent classes for the given ontology class.
        """
        return self.get_owl_items(owl_class.parents())

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose synonyms are to be retrieved.

        Returns:
            List: An empty list as no synonyms are implemented for this ontology class.
        """
        return []

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves comments for the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose comments are to be retrieved.

        Returns:
            List: An empty list as no comments are implemented for this ontology class.
        """
        return []

    def get_owl_classes(self, ontology: Any) -> Any:
        """
        Retrieves all classes from the ontology.

        Parameters:
            ontology (Any): The ontology whose classes are to be retrieved.

        Returns:
            Any: The classes of the ontology.
        """
        return ontology.all_classes

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Loads an ontology from the specified file.

        Parameters:
            input_file_path (str): The path to the ontology file to be loaded.

        Returns:
            Any: The loaded ontology.
        """
        ontology = ontospy.Ontospy(input_file_path, verbose=True)
        return ontology


class MatOntoOntology(BaseOntologyParser):
    """
    A parser for the MatOnto ontology, which provides methods for extracting comments and synonyms.
    """
    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves comments for the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose comments are to be retrieved.

        Returns:
            List: A list of comments for the ontology class.
        """
        return owl_class.comment.en

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for the ontology class.

        Parameters:
            owl_class (Any): The ontology class whose synonyms are to be retrieved.

        Returns:
            List: A list of synonyms for the ontology class.
        """
        return owl_class.synonym


class MaterialInformationEMMOOMDataset(OMDataset):
    """
    A dataset class for working with the MaterialInformation-EMMO ontology.

    This class maps the source ontology `MaterialInformationOntoOntology` to the target
    ontology `EMMOOntology` for the MaterialInformation-EMMO dataset.
    """
    track = track
    ontology_name = "MaterialInformation-EMMO"

    source_ontology = MaterialInformationOntoOntology()
    target_ontology = EMMOOntology()


class MaterialInformationMatOntoMDataset(OMDataset):
    """
    A dataset class for working with the MaterialInformation-MatOnto ontology.

    This class maps the source ontology `MaterialInformationOntoOntology` to the target
    ontology `MatOntoOntology` for the MaterialInformation-MatOnto dataset.
    """
    track = track
    ontology_name = "MaterialInformation-MatOnto"

    source_ontology = MaterialInformationOntoOntology()
    target_ontology = MatOntoOntology()


class MaterialInformationMatOntoReducedMDataset(OMDataset):
    """
    A dataset class for working with the MaterialInformationReduced-MatOnto ontology.

    This class maps the source ontology `MaterialInformationOntoOntology` to the target
    ontology `MatOntoOntology` for the MaterialInformationReduced-MatOnto dataset.
    """
    track = track
    ontology_name = "MaterialInformationReduced-MatOnto"

    source_ontology = MaterialInformationOntoOntology()
    target_ontology = MatOntoOntology()

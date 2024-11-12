# -*- coding: utf-8 -*-
"""
This script defines ontology parsers for the Mouse and Human ontologies, extending the base ontology parser.
It also defines a dataset class `MouseHumanOMDataset` that combines both ontologies for a specific task.

Classes:
- MouseOntology: Parses the Mouse ontology, specifically extracting comments for each ontology class.
- HumanOntology: Parses the Human ontology, specifically extracting definitions for each ontology class.
- MouseHumanOMDataset: A dataset class that combines the Mouse and Human ontologies for use in a specific task.

Each class inherits from the base ontology parser, which provides methods for extracting various types of information from the ontologies.
"""
import os.path
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "anatomy"


class MouseOntology(BaseOntologyParser):
    """
    This class extends the `BaseOntologyParser` to parse the Mouse ontology.

    It overrides the `get_comments` method to extract comments from the ontology classes.

    Methods:
        get_comments(self, owl_class: Any) -> List:
            Retrieve the comments for a given class in the Mouse ontology.
    """
    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieve the comments for a given class in the Mouse ontology.

        Parameters:
            owl_class (Any): An individual class from the Mouse ontology.

        Returns:
            List: A list of comments associated with the given ontology class.
        """
        return self.get_owl_items(owl_class.comment)


class HumanOntology(BaseOntologyParser):
    """
    This class extends the `BaseOntologyParser` to parse the Human ontology.

    It overrides the `get_comments` method to extract definitions for the ontology classes.

    Methods:
        get_comments(self, owl_class: Any) -> List:
            Retrieve the definitions for a given class in the Human ontology.
    """
    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieve the definitions for a given class in the Human ontology.

        Parameters:
            owl_class (Any): An individual class from the Human ontology.

        Returns:
            List: A list of definitions associated with the given ontology class.
        """
        return self.get_owl_items(owl_class.hasDefinition)


class MouseHumanOMDataset(OMDataset):
    """
    A dataset class that combines both the Mouse and Human ontologies for use in a specific task.

    This class specifies the source and target ontologies (Mouse and Human, respectively),
    and defines the working directory for the dataset. The class also associates it with the
    "anatomy" track.

    Attributes:
        track (str): The track for the dataset, set to "anatomy".
        ontology_name (str): The name of the ontology, set to "mouse-human".
        source_ontology (MouseOntology): The Mouse ontology parser instance.
        target_ontology (HumanOntology): The Human ontology parser instance.
        working_dir (str): The directory path for the working directory, combining track and ontology name.
    """
    track = track
    ontology_name = "mouse-human"

    source_ontology = MouseOntology()
    target_ontology = HumanOntology()

    working_dir = os.path.join(track, ontology_name)

# -*- coding: utf-8 -*-
"""
This script defines classes for parsing and processing datasets that are based on the
CommonKG (Common Knowledge Graph) ontology. The main objective is to provide ontology
parsers and dataset configurations for specific datasets (e.g., Nell-DBpedia, Yago-Wikidata).

The script includes the following components:
- `CommonKGOntology`: A class to handle the parsing of CommonKG ontology, with methods
  for extracting comments and synonyms (though these methods currently return empty lists).
- Dataset classes (`NellDbpediaOMDataset`, `YagoWikidataOMDataset`): These classes combine
  source and target ontologies with the `OMDataset` base class to create datasets using
  specific ontology configurations.

Classes:
- `CommonKGOntology`: A parser for the CommonKG ontology.
- `NellDbpediaOMDataset`: A dataset combining Nell and DBpedia ontologies.
- `YagoWikidataOMDataset`: A dataset combining Yago and Wikidata ontologies.

"""

import os.path
from typing import Any, List

from ..base import BaseOntologyParser, OMDataset

track = "commonkg"


class CommonKGOntology(BaseOntologyParser):
    """
    A class for parsing and handling the CommonKG ontology.

    This class provides methods to retrieve comments and synonyms for ontology classes,
    though in its current form, these methods return empty lists.

    Methods:
        get_comments(owl_class: Any) -> List: Returns an empty list for comments of the given ontology class.
        get_synonyms(owl_class: Any) -> List: Returns an empty list for synonyms of the given ontology class.
    """
    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves the comments associated with the given ontology class.

        In this implementation, it currently returns an empty list.

        Parameters:
            owl_class (Any): The ontology class whose comments are to be retrieved.

        Returns:
            List: An empty list, as this method is not yet implemented.
        """
        return []

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves the synonyms associated with the given ontology class.

        In this implementation, it currently returns an empty list.

        Parameters:
            owl_class (Any): The ontology class whose synonyms are to be retrieved.

        Returns:
            List: An empty list, as this method is not yet implemented.
        """
        return []


class NellDbpediaOMDataset(OMDataset):
    """
    A dataset class combining the Nell and DBpedia ontologies for knowledge graph data.

    This class uses the `CommonKGOntology` for both the source and target ontologies
    and defines the working directory for the dataset.

    Attributes:
        track (str): The track associated with this dataset, set to "commonkg".
        ontology_name (str): The name of the dataset, set to "nell-dbpedia".
        source_ontology (CommonKGOntology): The source ontology parser for Nell.
        target_ontology (CommonKGOntology): The target ontology parser for DBpedia.
        working_dir (str): The directory where the dataset files are stored, based on the track and ontology name.
    """
    track = track
    ontology_name = "nell-dbpedia"
    source_ontology = CommonKGOntology()
    target_ontology = CommonKGOntology()
    working_dir = os.path.join(track, ontology_name)


class YagoWikidataOMDataset(OMDataset):
    """
    A dataset class combining the Yago and Wikidata ontologies for knowledge graph data.

    This class uses the `CommonKGOntology` for both the source and target ontologies
    and defines the working directory for the dataset.

    Attributes:
        track (str): The track associated with this dataset, set to "commonkg".
        ontology_name (str): The name of the dataset, set to "yago-wikidata".
        source_ontology (CommonKGOntology): The source ontology parser for Yago.
        target_ontology (CommonKGOntology): The target ontology parser for Wikidata.
        working_dir (str): The directory where the dataset files are stored, based on the track and ontology name.
    """
    track = track
    ontology_name = "yago-wikidata"
    source_ontology = CommonKGOntology()
    target_ontology = CommonKGOntology()
    working_dir = os.path.join(track, ontology_name)

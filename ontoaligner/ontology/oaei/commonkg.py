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
This script defines classes for parsing and processing datasets that are based on the
CommonKG (Common Knowledge Graph) ontology. The main objective is to provide ontology
parsers and dataset configurations for specific datasets (e.g., Nell-DBpedia, Yago-Wikidata).
"""

import os.path
from typing import Any, List

from ...base import BaseOntologyParser, OMDataset

track = "commonkg"


class CommonKGOntology(BaseOntologyParser):
    """
    A class for parsing and handling the CommonKG ontology.

    This class provides methods to retrieve comments and synonyms for ontology classes,
    though in its current form, these methods return empty lists.
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
    """
    track = track
    ontology_name = "yago-wikidata"
    source_ontology = CommonKGOntology()
    target_ontology = CommonKGOntology()
    working_dir = os.path.join(track, ontology_name)

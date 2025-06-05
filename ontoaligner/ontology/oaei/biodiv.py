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
This script defines ontology parsers for various ontologies (ENVO, SWEET, SeaLife, TAXREFLD, NCBI)
and creates specific dataset classes for each ontology pairing. The ontology parsers extract various
types of information such as synonyms, comments, and labels for different biological and environmental
categories.
"""

import os.path
from typing import Any, List

from ...base import BaseOntologyParser, OMDataset

track = "biodiv"


class EnvoOntology(BaseOntologyParser):
    """
    This class extends `BaseOntologyParser` to parse the ENVO (Environmental Ontology).
    """
    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves comments for a given class in the ENVO ontology.

        Parameters:
            owl_class (Any): An ontology class from the ENVO ontology.

        Returns:
            List: A list of comments associated with the given ontology class.
        """
        return owl_class.comment

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for a given class in the ENVO ontology.

        Parameters:
            owl_class (Any): An ontology class from the ENVO ontology.

        Returns:
            List: A list of synonyms associated with the given ontology class.
        """
        return owl_class.hasRelatedSynonym


class SweetOntology(BaseOntologyParser):
    """
    This class extends `BaseOntologyParser` to parse the SWEET ontology (Sweet Earth Ontology).
    """
    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if a given class in the SWEET ontology contains a valid label.

        Parameters:
            owl_class (Any): An ontology class from the SWEET ontology.

        Returns:
            bool: True if the class contains a valid label, False otherwise.
        """
        try:
            if owl_class.name == "Thing":
                return False
            if len(owl_class.prefixIRI) == 0:
                return False
            return True
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves the label for a given class in the SWEET ontology.

        Parameters:
            owl_class (Any): An ontology class from the SWEET ontology.

        Returns:
            str: The label of the ontology class.
        """
        return str(owl_class.prefixIRI.first())

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves an empty list of comments for the SWEET ontology.

        Parameters:
            owl_class (Any): An ontology class from the SWEET ontology.

        Returns:
            List: An empty list, as SWEET ontology does not contain comments.
        """
        return []

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves an empty list of synonyms for the SWEET ontology.

        Parameters:
            owl_class (Any): An ontology class from the SWEET ontology.

        Returns:
            List: An empty list, as SWEET ontology does not contain synonyms.
        """
        return []


class EnvoSweetOMDataset(OMDataset):
    """
    Dataset combining the ENVO and SWEET ontologies.

    This dataset includes the necessary ontologies for source and target data
    and defines the working directory for this specific dataset.
    """
    track = track
    ontology_name = "envo-sweet"
    source_ontology = EnvoOntology()
    target_ontology = SweetOntology()
    working_dir = os.path.join(track, ontology_name)


class SeaLifeOntology(BaseOntologyParser):
    """
    This class extends `BaseOntologyParser` to parse the SeaLife ontology.
    """
    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if a given class in the SeaLife ontology contains a label.

        Parameters:
            owl_class (Any): An ontology class from the SeaLife ontology.

        Returns:
            bool: True if the class contains a label, False otherwise.
        """
        if len(owl_class.label.en) == 0:
            return False
        return True

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves the label for a given class in the SeaLife ontology.

        Parameters:
            owl_class (Any): An ontology class from the SeaLife ontology.

        Returns:
            str: The label of the ontology class.
        """
        return str(owl_class.label.en.first())

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for a given class in the SeaLife ontology.

        Parameters:
            owl_class (Any): An ontology class from the SeaLife ontology.

        Returns:
            List: A list of synonyms associated with the given ontology class.
        """
        return owl_class.hasRelatedSynonym.en

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves comments for a given class in the SeaLife ontology.

        Parameters:
            owl_class (Any): An ontology class from the SeaLife ontology.

        Returns:
            List: A list of comments associated with the given ontology class.
        """
        return owl_class.comment.en


class FishZooplanktonOMDataset(OMDataset):
    """
    Dataset combining the SeaLife ontology for both source and target.

    This dataset includes the SeaLife ontology as both the source and target ontology
    and defines the working directory for this dataset.
    """
    track = track
    ontology_name = "fish-zooplankton"
    source_ontology = SeaLifeOntology()
    target_ontology = SeaLifeOntology()
    working_dir = os.path.join(track, ontology_name)


class MacroalgaeMacrozoobenthosOMDataset(OMDataset):
    """
    Dataset combining the SeaLife ontology for both source and target, specific
    to macroalgae and macrozoobenthos categories.
    """
    track = track
    ontology_name = "macroalgae-macrozoobenthos"
    source_ontology = SeaLifeOntology()
    target_ontology = SeaLifeOntology()
    working_dir = os.path.join(track, ontology_name)


class TAXREFLDOntology(BaseOntologyParser):
    """
    Dataset combining the SeaLife ontology for both source and target, specific
    to macroalgae and macrozoobenthos categories.

    Attributes:
        track (str): The track associated with the dataset, set to "biodiv".
        ontology_name (str): The name of the ontology dataset, set to "macroalgae-macrozoobenthos".
        source_ontology (SeaLifeOntology): The source ontology parser (SeaLife).
        target_ontology (SeaLifeOntology): The target ontology parser (SeaLife).
        working_dir (str): The directory where the dataset files are stored, based on the track and ontology name.
    """
    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for a given class in the TAXREFLD ontology.

        Parameters:
            owl_class (Any): An ontology class from the TAXREFLD ontology.

        Returns:
            List: A list of synonyms associated with the given ontology class.
        """
        return owl_class.hasSynonym

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves comments for a given class in the TAXREFLD ontology.

        Parameters:
            owl_class (Any): An ontology class from the TAXREFLD ontology.

        Returns:
            List: A list of comments associated with the given ontology class.
        """
        return owl_class.comment.en


class NCBIOntology(BaseOntologyParser):
    """
    This class extends `BaseOntologyParser` to parse the NCBI Taxonomy ontology,
    which provides taxonomic information on a wide range of organisms.
    """
    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves synonyms for a given class in the NCBI ontology.

        Parameters:
            owl_class (Any): An ontology class from the NCBI ontology.

        Returns:
            List: A list of synonyms associated with the given ontology class.
        """
        return owl_class.hasRelatedSynonym

    def get_comments(self, owl_class: Any) -> List:
        """
        Retrieves comments for a given class in the NCBI ontology.

        Parameters:
            owl_class (Any): An ontology class from the NCBI ontology.

        Returns:
            List: A list of comments associated with the given ontology class.
        """
        return []


class TaxrefldBacteriaNcbitaxonBacteriaOMDataset(OMDataset):
    """
    Dataset combining the TAXREFLD and NCBI ontologies for the Bacteria category.
    """
    track = track
    ontology_name = "taxrefldBacteria-ncbitaxonBacteria"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldChromistaNcbitaxonChromistaOMDataset(OMDataset):
    """
    Dataset combining the TAXREFLD and NCBI ontologies for the Chromista category.
    """
    track = track
    ontology_name = "taxrefldChromista-ncbitaxonChromista"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldFungiNcbitaxonFungiOMDataset(OMDataset):
    """
    Dataset combining the TAXREFLD and NCBI ontologies for the Fungi category.
    """
    track = track
    ontology_name = "taxrefldFungi-ncbitaxonFungi"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldPlantaeNcbitaxonPlantaeOMDataset(OMDataset):
    """
    Dataset combining the TAXREFLD and NCBI ontologies for the Plantae category.
    """
    track = track
    ontology_name = "taxrefldPlantae-ncbitaxonPlantae"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)


class TaxrefldProtozoaNcbitaxonProtozoaOMDataset(OMDataset):
    """
    Dataset combining the TAXREFLD and NCBI ontologies for the Protozoa category.
    """
    track = track
    ontology_name = "taxrefldProtozoa-ncbitaxonProtozoa"
    source_ontology = TAXREFLDOntology()
    target_ontology = NCBIOntology()
    working_dir = os.path.join(track, ontology_name)

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
This script defines ontology parsers for the Mouse and Human ontologies, extending the base ontology parser.
"""
import os.path
from typing import Any, List

from ...base import BaseOntologyParser, OMDataset

track = "anatomy"


class MouseOntology(BaseOntologyParser):
    """
    This class extends the `BaseOntologyParser` to parse the Mouse ontology.

    It overrides the `get_comments` method to extract comments from the ontology classes.
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
    """
    track = track
    ontology_name = "mouse-human"

    source_ontology = MouseOntology()
    target_ontology = HumanOntology()

    working_dir = os.path.join(track, ontology_name)

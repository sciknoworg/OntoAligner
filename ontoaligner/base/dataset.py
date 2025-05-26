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
The script is responsible for loading and collecting data related to source and target ontologies, along with reference alignments.
It provides methods for collecting data, loading from JSON, and handling file paths.

Classes:
    - OMDataset: A base class for handling ontology matching datasets, including parsing ontologies and alignments
                and collecting dataset-related information.
"""
import os
import json
from abc import ABC
from typing import Any, Dict

from ontoaligner.base.ontology import BaseAlignmentsParser



class OMDataset(ABC):
    """
    A base class for managing ontology matching datasets, including the source and target ontologies,
    and the reference alignments.

    This class is responsible for collecting ontology data, parsing ontologies, and handling file paths
    associated with the dataset. It provides methods for data collection, loading data from JSON files,
    and retrieving directory paths.

    Attributes:
        track (str): The dataset track name.
        ontology_name (str): The name of the ontology being processed.
        source_ontology (Any): The source ontology object.
        target_ontology (Any): The target ontology object.
        alignments (Any): The alignments parser object, using `BaseAlignmentsParser`.
    """

    track: str = ""
    ontology_name: str = ""

    source_ontology: Any = None
    target_ontology: Any = None

    alignments: Any = BaseAlignmentsParser()

    def collect(self, source_ontology_path: str, target_ontology_path: str, reference_matching_path: str="") -> Dict:
        """
        Collects data from the source ontology, target ontology, and reference alignments.

        This method takes paths to the source ontology, target ontology, and reference alignments files,
        parses them, and returns a dictionary containing the dataset information, source, target,
        and reference alignments data.

        Parameters:
            source_ontology_path (str): The file path to the source ontology.
            target_ontology_path (str): The file path to the target ontology.
            reference_matching_path (str): The file path to the reference matching alignments.

        Returns:
            Dict: A dictionary containing the dataset information, parsed source and target ontologies,
                  and parsed reference alignments.
        """
        data = {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source": self.source_ontology.parse(input_file_path=source_ontology_path),
            "target": self.target_ontology.parse(input_file_path=target_ontology_path),
            "reference": self.alignments.parse(input_file_path=reference_matching_path),
        }
        return data

    def load_from_json(self, json_file_path: str) -> Dict:
        """
        Loads dataset information from a JSON file.

        This method loads the dataset's information from a JSON file located at a specific path,
        constructed from the root directory, track, and ontology name.

        Parameters:
            root_dir (str): The root directory where the dataset's JSON file is located.

        Returns:
            Dict: The JSON data loaded from the specified file.
        """
        with open(json_file_path, encoding="utf-8") as f:
            json_data = json.load(f)
        return json_data

    def __dir__(self):
        """
        Returns the directory structure for the dataset.

        This method constructs the directory path based on the track and ontology name.

        Returns:
            str: The constructed directory path for the dataset.
        """
        return os.path.join(self.track, self.ontology_name)

    def __str__(self):
        """
        Returns a string representation of the dataset's ontology name.

        This method returns a string containing the ontology name.

        Returns:
            str: The ontology name as a string.
        """
        return f"{self.ontology_name}"

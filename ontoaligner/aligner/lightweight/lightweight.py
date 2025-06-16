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
This script defines models for ontology matching, specifically a lightweight model
and an extension that uses fuzzy string matching via the RapidFuzz library.

The `Lightweight` class serves as a base for a simple ontology matching model, while the
`FuzzySMLightweight` class enhances it with fuzzy string matching to calculate similarities
between source and target ontologies.

Classes:
    - Lightweight: A basic ontology matching model with placeholder methods.
    - FuzzySMLightweight: A subclass of `Lightweight` that incorporates fuzzy string matching using RapidFuzz.
"""
from typing import Any, List

import rapidfuzz
from tqdm import tqdm

from ...base import BaseOMModel


class Lightweight(BaseOMModel):
    """
    A lightweight ontology matching model that serves as a base for other models.

    This class does not load or process models but provides basic structure and methods
    for ontology matching. It includes methods for initializing a retriever and generating
    matching results, which can be extended in subclasses.
    """

    def __init__(self, fuzzy_sm_threshold: float=0.5, **kwargs) -> None:
        """
        Initializes the ontology matching model with optional keyword arguments.

        Parameters:
            fuzzy_sm_threshold(float): Contains the threshold value for fuzzy string matching (e.g., 'fuzzy_sm_threshold').
            **kwargs: Additional keyword arguments that may be used for model configuration or parameters.
        """
        kwargs['fuzzy_sm_threshold'] = fuzzy_sm_threshold
        super().__init__(**kwargs)

    def __str__(self):
        """
        Returns a string representation of the Lightweight model.

        Returns:
            str: A simple string representation of the class ("Lightweight").
        """
        return "Lightweight"

    def init_retriever(self, data):
        """
        Initializes a retriever for the model. In this class, it does nothing but can be
        extended in subclasses for actual implementation.

        Args:
            data (Any): The data required to initialize the retriever. Not used in this class.
        """
        pass

    def generate(self, input_data: List) -> List:
        """
        Generates matching results based on the input data. In this class, it is a placeholder
        method that does not perform any matching.

        Args:
            input_data (List): A list containing source and target ontologies for the matching task.

        Returns:
            List: An empty list as this method is not yet implemented in the base class.
        """
        # source_onto, target_onto
        # index into retriever
        pass


class FuzzySMLightweight(Lightweight):
    """
    Fuzzy String Matching using: https://github.com/maxbachmann/RapidFuzz#partial-ratio

    A subclass of `Lightweight` that uses fuzzy string matching for ontology matching using
    the RapidFuzz library.

    This class calculates similarity scores between source and target ontologies based on
    partial string matching. It filters results based on a predefined threshold and returns
    matching predictions.
    """

    def ratio_estimate(self) -> Any:
        """
        Estimates a ratio for fuzzy string matching. In this class, it is a placeholder method
        that can be customized to define how the ratio is calculated.

        Returns:
            Any: A placeholder return value, can be overridden by subclasses.
        """
        pass

    def calculate_similarity(self, source: str, candidates: List) -> [int, float]:
        """
        Calculates the similarity between a source string and a list of candidate strings
        using RapidFuzz's string matching capabilities.

        Args:
            source (str): The source string for matching.
            candidates (List): A list of candidate strings to match the source against.

        Returns:
            List: A list containing the index of the most similar candidate and the normalized similarity score.
        """
        selected_candid = rapidfuzz.process.extractOne(
            source,
            candidates,
            scorer=self.ratio_estimate(),
            processor=rapidfuzz.utils.default_process,
        )
        return selected_candid[2], selected_candid[1] / 100

    def generate(self, input_data: List) -> List:
        """
        Generates predictions by comparing source and target ontologies using fuzzy string matching.

        Args:
            input_data (List): A list containing source and target ontologies for matching. The first item
                                is the source ontology, and the second is the target ontology.

        Returns:
            List: A list of predictions, each containing the source and target ontology IRI along with the similarity score,
                  for matches above a certain threshold (`fuzzy_sm_threshold`).
        """
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []
        candidates = [target["text"] for target in target_ontology]
        for source in tqdm(source_ontology):
            selected_candid_idx, selected_candid_score = self.calculate_similarity(source=source["text"],
                                                                                   candidates=candidates)
            if selected_candid_score >= self.kwargs["fuzzy_sm_threshold"]:
                predictions.append(
                    {
                        "source": source["iri"],
                        "target": target_ontology[selected_candid_idx]["iri"],
                        "score": selected_candid_score,
                    }
                )
        return predictions

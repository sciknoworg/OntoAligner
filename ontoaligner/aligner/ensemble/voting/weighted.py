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
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .base import BaseVoting, _get_unique_sorted_predictions

class WeightedVoting(BaseVoting):
    """
    A weighted voting method for combining alignment predictions.

    This class combines branch predictions by counting weighted branch support for each
    source-target pair.
    """

    voting_method: str = "WeightedVoting"

    def __init__(self, min_votes: int = 1, score_threshold: float = None):
        """
        Initializes the weighted voting method.

        Parameters:
            min_votes (int, optional): Minimum number of branches required for a pair. Defaults to 1.
            score_threshold (float, optional): Minimum branch score required to count a vote. Defaults to None.
        """
        self.min_votes = min_votes
        self.score_threshold = score_threshold

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            dict: A dictionary with the class name as key and voting configuration as value.
        """
        return {
            "WeightedVoting": {
                "min_votes": self.min_votes,
                "score_threshold": self.score_threshold,
            }
        }

    def combine(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines branch predictions using weighted voting.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and branch weights.

        Returns:
            List[Dict]: A list of combined source-target predictions sorted by vote score.
        """
        vote_scores = defaultdict(float)
        vote_counts = defaultdict(int)

        for flat_predictions, weight in branch_outputs:
            unique_predictions = _get_unique_sorted_predictions(predictions=flat_predictions)

            for prediction in unique_predictions:
                score = float(prediction["score"])

                if self.score_threshold is not None and score < self.score_threshold:
                    continue

                pair = (prediction["source"], prediction["target"])
                vote_scores[pair] += float(weight)
                vote_counts[pair] += 1

        return [
            {"source": source, "target": target, "score": score}
            for (source, target), score in sorted(
                vote_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if vote_counts[(source, target)] >= self.min_votes
        ]

    def get_voting_info(self) -> Any:
        """
        Provides information about the voting method.

        Returns:
            str: A description of the voting method's function in the ensemble aligner.
        """
        return "INPUT CONSIST OF BRANCH PREDICTIONS TO WEIGHTED VOTING"

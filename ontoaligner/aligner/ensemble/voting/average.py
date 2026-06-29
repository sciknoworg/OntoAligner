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

class ScoreAverageVoting(BaseVoting):
    """
    A score averaging voting method for combining alignment predictions.

    This class combines branch predictions by averaging source-target scores across
    branches using branch weights.
    """

    voting_method: str = "ScoreAverageVoting"

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            str: A simple string representation of the class ("ScoreAverageVoting").
        """
        return "ScoreAverageVoting"

    def combine(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines branch predictions using weighted score averaging.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and branch weights.

        Returns:
            List[Dict]: A list of combined source-target predictions sorted by averaged score.
        """
        score_sums = defaultdict(float)
        weight_sums = defaultdict(float)

        for flat_predictions, weight in branch_outputs:
            unique_predictions = _get_unique_sorted_predictions(predictions=flat_predictions)
            branch_weight = float(weight)

            for prediction in unique_predictions:
                pair = (prediction["source"], prediction["target"])
                score_sums[pair] += branch_weight * float(prediction["score"])
                weight_sums[pair] += branch_weight

        return [
            {
                "source": source,
                "target": target,
                "score": score_sums[(source, target)] / weight_sums[(source, target)],
            }
            for (source, target) in sorted(
                score_sums.keys(),
                key=lambda pair: score_sums[pair] / weight_sums[pair],
                reverse=True,
            )
            if weight_sums[(source, target)] != 0
        ]

    def get_voting_info(self) -> Any:
        """
        Provides information about the voting method.

        Returns:
            str: A description of the voting method's function in the ensemble aligner.
        """
        return "INPUT CONSIST OF SCORED BRANCH PREDICTIONS TO SCORE AVERAGE VOTING"

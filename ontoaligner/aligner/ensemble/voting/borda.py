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


class BordaCountVoting(BaseVoting):
    """
    A Borda count voting method for combining alignment predictions.

    This class combines branch predictions by assigning higher scores to higher-ranked
    source-target pairs in each branch.
    """

    voting_method: str = "BordaCountVoting"

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            str: A simple string representation of the class ("BordaCountVoting").
        """
        return "BordaCountVoting"

    def combine(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines branch predictions using Borda count voting.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and branch weights.

        Returns:
            List[Dict]: A list of combined source-target predictions sorted by fused score.
        """
        fused_scores = defaultdict(float)

        for flat_predictions, weight in branch_outputs:
            sorted_predictions = _get_unique_sorted_predictions(predictions=flat_predictions)
            total_predictions = len(sorted_predictions)

            if total_predictions == 0:
                continue

            for rank, prediction in enumerate(sorted_predictions, start=1):
                pair = (prediction["source"], prediction["target"])
                borda_score = (total_predictions - rank + 1) / total_predictions
                fused_scores[pair] += float(weight) * borda_score

        return [
            {"source": source, "target": target, "score": score}
            for (source, target), score in sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    def get_voting_info(self) -> Any:
        """
        Provides information about the voting method.

        Returns:
            str: A description of the voting method's function in the ensemble aligner.
        """
        return "INPUT CONSIST OF RANKED BRANCH PREDICTIONS TO BORDA COUNT VOTING"

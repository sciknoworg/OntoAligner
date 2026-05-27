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


class ReciprocalRankFusionVoting(BaseVoting):
    """
    A reciprocal rank fusion voting method for combining alignment predictions.

    This class combines branch predictions by ranking each branch output independently
    and adding reciprocal-rank scores for repeated source-target pairs.

    Attributes:
        voting_method (str): The name of the voting method.
    """

    voting_method: str = "ReciprocalRankFusion"

    def __init__(self, k: int = 60):
        """
        Initializes the ReciprocalRankFusion voting method.

        Parameters:
            k (int, optional): Smoothing constant used in reciprocal rank fusion. Defaults to 60.
        """
        self.k = k

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            dict: A dictionary with the class name as key and voting configuration as value.
        """
        return {"ReciprocalRankFusion": {"k": self.k}}

    def combine(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines branch predictions using reciprocal rank fusion.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and branch weights.

        Returns:
            List[Dict]: A list of combined source-target predictions sorted by fused score.
        """
        fused_scores = defaultdict(float)

        for flat_predictions, weight in branch_outputs:
            sorted_predictions = _get_unique_sorted_predictions(predictions=flat_predictions)

            for rank, prediction in enumerate(sorted_predictions, start=1):
                pair = (prediction["source"], prediction["target"])
                fused_scores[pair] += float(weight) / (self.k + rank)

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
        return "INPUT CONSIST OF RANKED BRANCH PREDICTIONS TO RECIPROCAL RANK FUSION"

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


class CondorcetVoting(BaseVoting):
    """
    A Condorcet voting method for combining alignment predictions.

    This class compares target candidates pairwise for each source entity and ranks
    candidates by the number of pairwise victories.
    """

    voting_method: str = "CondorcetVoting"

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            str: A simple string representation of the class ("CondorcetVoting").
        """
        return "CondorcetVoting"

    def _get_source_candidates(self, branch_outputs: List[Tuple[List[Dict], float]]) -> Dict:
        """
        Collects target candidates for each source entity.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and branch weights.

        Returns:
            Dict: A mapping from source IRI to target candidate IRIs.
        """
        source_candidates = defaultdict(set)

        for flat_predictions, _ in branch_outputs:
            for prediction in flat_predictions:
                source_candidates[prediction["source"]].add(prediction["target"])

        return source_candidates

    def _get_branch_rankings(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Tuple[Dict, float]]:
        """
        Builds source-target rank maps for each branch.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and branch weights.

        Returns:
            List[Tuple[Dict, float]]: A list of branch ranking maps and branch weights.
        """
        branch_rankings = []

        for flat_predictions, weight in branch_outputs:
            sorted_predictions = _get_unique_sorted_predictions(predictions=flat_predictions)
            source_rankings = defaultdict(dict)

            for rank, prediction in enumerate(sorted_predictions, start=1):
                source = prediction["source"]
                target = prediction["target"]

                if target not in source_rankings[source]:
                    source_rankings[source][target] = rank

            branch_rankings.append((source_rankings, float(weight)))

        return branch_rankings

    def combine(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines branch predictions using Condorcet voting.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and branch weights.

        Returns:
            List[Dict]: A list of combined source-target predictions sorted by Condorcet score.
        """
        source_candidates = self._get_source_candidates(branch_outputs=branch_outputs)
        branch_rankings = self._get_branch_rankings(branch_outputs=branch_outputs)

        condorcet_scores = defaultdict(float)

        for source, candidates in source_candidates.items():
            candidates = list(candidates)

            for target in candidates:
                pair = (source, target)

                for opponent in candidates:
                    if target == opponent:
                        continue

                    wins = 0.0
                    losses = 0.0

                    for source_rankings, weight in branch_rankings:
                        source_ranks = source_rankings.get(source, {})
                        missing_rank = len(source_ranks) + 1

                        target_rank = source_ranks.get(target, missing_rank)
                        opponent_rank = source_ranks.get(opponent, missing_rank)

                        if target_rank < opponent_rank:
                            wins += weight
                        elif target_rank > opponent_rank:
                            losses += weight

                    if wins > losses:
                        condorcet_scores[pair] += 1.0
                    elif wins == losses:
                        condorcet_scores[pair] += 0.5

        return [
            {"source": source, "target": target, "score": score}
            for (source, target), score in sorted(
                condorcet_scores.items(),
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
        return "INPUT CONSIST OF RANKED BRANCH PREDICTIONS TO CONDORCET VOTING"

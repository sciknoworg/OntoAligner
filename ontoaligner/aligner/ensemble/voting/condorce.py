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

    def __init__(self, selection: str = "top1_source", threshold: float = None, top_k: int = None, margin: float = None):
        """
        Initializes the CondorcetVoting voting method.
        """
        super().__init__(selection=selection, threshold=threshold, top_k=top_k, margin=margin)

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            str: A simple string representation of the class ("CondorcetVoting").
        """
        return "CondorcetVoting"

    def _get_source_candidates(self, aligner_outputs: List[Tuple[List[Dict], float]]) -> Dict:
        """
        Collects target candidates for each source entity.

        Parameters:
            aligner_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and aligner weights.

        Returns:
            Dict: A mapping from source IRI to target candidate IRIs.
        """
        source_candidates = defaultdict(set)

        for flat_predictions, _ in aligner_outputs:
            for prediction in flat_predictions:
                source_candidates[prediction["source"]].add(prediction["target"])

        return source_candidates

    def _get_aligner_rankings(self, aligner_outputs: List[Tuple[List[Dict], float]]) -> List[Tuple[Dict, float]]:
        """
        Builds source-target rank maps for each aligner.

        Parameters:
            aligner_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and aligner weights.

        Returns:
            List[Tuple[Dict, float]]: A list of aligner ranking maps and aligner weights.
        """
        aligner_rankings = []

        for flat_predictions, weight in aligner_outputs:
            sorted_predictions = _get_unique_sorted_predictions(predictions=flat_predictions)
            source_rankings = defaultdict(dict)

            for rank, prediction in enumerate(sorted_predictions, start=1):
                source = prediction["source"]
                target = prediction["target"]

                if target not in source_rankings[source]:
                    source_rankings[source][target] = rank

            aligner_rankings.append((source_rankings, float(weight)))

        return aligner_rankings

    def fuse(self, aligner_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines aligner predictions using Condorcet voting.

        Parameters:
            aligner_outputs (List[Tuple[List[Dict], float]]): A list of flat predictions and aligner weights.

        Returns:
            List[Dict]: A list of combined source-target predictions sorted by Condorcet score.
        """
        source_candidates = self._get_source_candidates(aligner_outputs=aligner_outputs)
        aligner_rankings = self._get_aligner_rankings(aligner_outputs=aligner_outputs)

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

                    for source_rankings, weight in aligner_rankings:
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

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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from collections import defaultdict


def _get_unique_sorted_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Sorts predictions by score and removes duplicate source-target pairs.

    Parameters:
        predictions (List[Dict]): A list of flat source-target-score predictions.

    Returns:
        List[Dict]: A sorted list of unique source-target-score predictions.
    """
    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: float(prediction["score"]),
        reverse=True,
    )

    unique_predictions = []
    seen_pairs = set()

    for prediction in sorted_predictions:
        pair = (prediction["source"], prediction["target"])

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)
        unique_predictions.append(prediction)

    return unique_predictions


class BaseVoting(ABC):
    """
    An abstract base class for ensemble voting methods.

    This class defines the common interface used by voting strategies that combine
    predictions from multiple alignment branches.

    Attributes:
        voting_method (str): The name of the voting method.
    """

    voting_method: str = ""

    SELECTION_STRATEGIES = ("none", "top1_source", "bijective", "threshold", "topk_source")

    def __init__(self, selection: str = "none", threshold: float = None, top_k: int = None, margin: float = None):
        self.selection = selection
        self.threshold = threshold
        self.top_k = top_k
        self.margin = margin
        if selection not in self.SELECTION_STRATEGIES:
            raise ValueError(f"Unknown selection strategy '{selection}'. Must be one of {self.SELECTION_STRATEGIES}.")
        if selection == "threshold" and threshold is None:
            raise ValueError("threshold must be provided when selection='threshold'.")
        if selection == "topk_source" and top_k is None and margin is None:
            raise ValueError("At least one of top_k or margin must be provided when selection='topk_source'.")

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            str: The voting method name.
        """
        return self.voting_method

    def combine(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines predictions from multiple alignment branches.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat branch predictions and branch weights.

        Returns:
            List[Dict]: The combined source-target predictions.
        """
        fused = self.fuse(branch_outputs=branch_outputs)
        return self._select(fused=fused)

    @abstractmethod
    def fuse(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        pass

    @abstractmethod
    def get_voting_info(self) -> Any:
        """
        Provides information about the voting method.

        Returns:
            Any: A description of the voting method's function in the ensemble aligner.
        """
        pass

    def _select(self, fused: List[Dict]) -> List[Dict]:
        """
        Applies the configured post-fusion selection strategy to a fused, score-sorted
        prediction list, converting a ranked candidate pool into a decided alignment.

        Parameters:
            fused (List[Dict]): Fused predictions, sorted descending by score.

        Returns:
            List[Dict]: The filtered/selected predictions.
        """
        if self.selection == "top1_source":
            return self._select_top1_per_source(fused=fused)
        if self.selection == "bijective":
            return self._select_greedy_bijective(fused=fused)
        if self.selection == "threshold":
            return [pred for pred in fused if pred["score"] >= self.threshold]
        if self.selection == "topk_source":
            return self._select_topk_per_source(fused=fused, k=self.top_k, margin=self.margin)
        raise ValueError(f"Unknown selection strategy '{self.selection}'.")

    @staticmethod
    def _select_top1_per_source(fused: List[Dict]) -> List[Dict]:
        """
        Keeps only the highest-scoring target for each source entity.

        Parameters:
            fused (List[Dict]): Fused predictions, sorted descending by score.

        Returns:
            List[Dict]: One prediction per source entity.
        """
        best_per_source: Dict[str, Dict] = {}
        for prediction in fused:
            source = prediction["source"]
            if source not in best_per_source:
                best_per_source[source] = prediction
        return list(best_per_source.values())

    @staticmethod
    def _select_greedy_bijective(fused: List[Dict]) -> List[Dict]:
        """
        Greedily selects pairs such that each source and each target is used at most once.

        Parameters:
            fused (List[Dict]): Fused predictions, sorted descending by score.

        Returns:
            List[Dict]: A 1-to-1 subset of predictions.
        """
        used_sources, used_targets, selected = set(), set(), []
        for prediction in fused:
            source, target = prediction["source"], prediction["target"]
            if source in used_sources or target in used_targets:
                continue
            used_sources.add(source)
            used_targets.add(target)
            selected.append(prediction)
        return selected

    @staticmethod
    def _select_topk_per_source(fused: List[Dict], k: int = None, margin: float = None) -> List[Dict]:
        """
        Keeps, for each source, either its top-k targets or all targets within a relative
        score margin of its best target — whichever constraint is supplied. This preserves
        legitimate one-to-many correspondences instead of forcing a single target per source.

        Parameters:
            fused (List[Dict]): Fused predictions, sorted descending by score.
            k (int, optional): Max targets to keep per source. None means no cap.
            margin (float, optional): Keep targets whose score >= margin * best_score_for_source.
                                    E.g. margin=0.9 keeps anything within 10% of the top score.
                                    None means no margin filtering.

        Returns:
            List[Dict]: Selected predictions, possibly multiple per source.
        """
        if k is None and margin is None:
            raise ValueError("At least one of k or margin must be provided.")

        by_source = defaultdict(list)
        for prediction in fused:
            by_source[prediction["source"]].append(prediction)

        selected = []
        for source, preds in by_source.items():
            # preds already sorted desc because `fused` is sorted desc
            best_score = preds[0]["score"]
            kept = preds
            if margin is not None:
                kept = [p for p in kept if p["score"] >= margin * best_score]
            if k is not None:
                kept = kept[:k]
            selected.extend(kept)

        return selected

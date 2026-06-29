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

    def __str__(self):
        """
        Returns a string representation of the voting method.

        Returns:
            str: The voting method name.
        """
        return self.voting_method

    @abstractmethod
    def combine(self, branch_outputs: List[Tuple[List[Dict], float]]) -> List[Dict]:
        """
        Combines predictions from multiple alignment branches.

        Parameters:
            branch_outputs (List[Tuple[List[Dict], float]]): A list of flat branch predictions and branch weights.

        Returns:
            List[Dict]: The combined source-target predictions.
        """
        pass

    @abstractmethod
    def get_voting_info(self) -> Any:
        """
        Provides information about the voting method.

        Returns:
            Any: A description of the voting method's function in the ensemble aligner.
        """
        pass

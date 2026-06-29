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
from typing import Dict, List, Tuple

from ...base import BaseOMModel
from .voting import ReciprocalRankFusionVoting
from .voting.base import BaseVoting

class EnsembleLearningAligner(BaseOMModel):
    """
    An ensemble ontology matching aligner for combining predictions from multiple aligner pipelines.
    """

    def __init__(
        self,
        branches: List[Tuple],
        voting: BaseVoting = None,
        **kwargs,
    ) -> None:
        """
        Initializes the ensemble aligner.

        Parameters:
            branches (List[Tuple]): A list of branch tuples in the form
                                    (name, aligner_pipeline) or (name, aligner_pipeline, weight).
            voting (BaseVoting, optional): Voting method used to combine branch predictions.
                                        Defaults to ReciprocalRankFusionVoting.
            **kwargs: Additional keyword arguments that may be used for model configuration.
        """
        super().__init__(**kwargs)

        if len(branches) < 2:
            raise ValueError("EnsembleLearningAligner requires two or more aligner pipelines.")

        self.branches = []

        for branch in branches:
            if len(branch) == 2:
                name, aligner_pipeline = branch
                weight = 1.0
            elif len(branch) == 3:
                name, aligner_pipeline, weight = branch
            else:
                raise ValueError("Each branch must be (name, aligner_pipeline) or (name, aligner_pipeline, weight).")

            self.branches.append((name, aligner_pipeline, float(weight)))

        self.voting = voting or ReciprocalRankFusionVoting()

    def __str__(self):
        """
        Returns a string representation of the EnsembleLearningAligner model.

        Returns:
            str: A simple string representation of the class ("EnsembleLearningAligner").
        """
        return "EnsembleLearningAligner"

    def _flatten_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Flattens flat and grouped retrieval-style predictions into source-target predictions.

        Parameters:
            predictions (List[Dict]): A list of flat or grouped prediction dictionaries.

        Returns:
            List[Dict]: A flat list of source-target-score prediction dictionaries.
        """
        flat_predictions = []

        if predictions is None:
            return flat_predictions

        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise ValueError("Each prediction must be a dictionary.")

            if "target-cands" in prediction and "score-cands" in prediction:
                for target, score in zip(
                    prediction["target-cands"],
                    prediction["score-cands"],
                ):
                    flat_predictions.append(
                        {
                            "source": prediction["source"],
                            "target": target,
                            "score": float(score),
                        }
                    )

            elif "source" in prediction and "target" in prediction:
                flat_predictions.append(
                    {
                        "source": prediction["source"],
                        "target": prediction["target"],
                        "score": float(prediction.get("score", 1.0)),
                    }
                )

            else:
                raise ValueError(
                    "Predictions must be flat source-target dictionaries, "
                    "flat source-target-score dictionaries, or grouped retrieval-style "
                    "dictionaries with target-cands and score-cands."
                )

        return flat_predictions

    def generate(self, input_data: Dict = None) -> List:
        """
        Generates ensemble predictions by combining branch predictions.

        Parameters:
            input_data (Dict, optional): Optional ontology matching dataset forwarded to each branch.
                                         If not provided, each branch uses its own dataset.

        Returns:
            List: A list of fused source-target predictions sorted by score.
        """
        branch_outputs = []

        for _, aligner_pipeline, weight in self.branches:
            predictions = aligner_pipeline.generate(input_data=input_data)
            flat_predictions = self._flatten_predictions(predictions=predictions)
            branch_outputs.append((flat_predictions, weight))

        return self.voting.combine(branch_outputs=branch_outputs)

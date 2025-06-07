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
This script defines functions for evaluating the intersection between predicted and reference data,
as well as calculating various evaluation metrics such as precision, recall, and F-score.

It includes two main functions:

1. `calculate_intersection`: Computes the number of matching items between the predicted and reference data.

2. `evaluation_report`: Calculates precision, recall, and F-score based on the intersection of predicted and reference data.
"""

from typing import Dict, List

def calculate_intersection(predicts: List, references: List) -> int:
    """
        Calculate Matching Items Between Predicted and Reference Data:

        This function compares the predicted data with the reference data and determines the number of matching items.
        A match is identified when both the `source` and `target` fields are identical in a predicted-reference pair.

        Parameters:
        ------------

        - **`predicts`** (*list of dict*):
          A list of predicted entries, where each entry is a dictionary containing:
          - `source` (*any*): The source element of the prediction.
          - `target` (*any*): The target element of the prediction.
          - `score` (*float*): An optional confidence or relevance score for the prediction.

        - **`references`** (*list of dict*):
          A list of reference entries, where each entry is a dictionary containing:
          - `source` (*any*): The source element of the reference.
          - `target` (*any*): The target element of the reference.
          - `relation` (*any*): The relationship between the source and target in the reference.

        Returns:
        ---------

        - **`intersection`** (*list of dict*):
          A list of matching items, where each matching item is a dictionary containing the `source` and `target` fields
          found in both `predicts` and `references`.
    """
    intersection = 0
    for predict in predicts:
        for reference in references:
            if predict["source"] == reference["source"] and predict["target"] == reference["target"]:
                intersection += 1
                break
    return intersection


def evaluation_report(predicts: List, references: List, beta: int = 1) -> Dict:
    """
        Calculate Evaluation Metrics:

        - **Precision (P)**:
            - $$\( P = \frac{|\text{intersection of predicts and references}|}{|\text{predicts}|} \)$$
            - Measures the proportion of predicted elements that are correct.

        - **Recall (R)**:
            - $$\( R = \frac{|\text{intersection of predicts and references}|}{|\text{references}|} \)$$
            - Measures the proportion of reference elements that are successfully predicted.

        - **F-Score (F_β)**:
            - $$\( F_β = \frac{(1 + β^2) \cdot P \cdot R}{(β^2 \cdot P) + R} \)$$
            - A weighted harmonic mean of precision and recall, where \( β \) determines the balance between them.

        Parameters:
        ---------------

        - **`predicts`** (*list of dict*):
          A list of prediction entries, where each entry is a dictionary with the following keys:
          - `source` (*any*): The source element of the prediction.
          - `target` (*any*): The target element of the prediction.
          - `score` (*float*): The associated confidence or relevance score for the prediction.

        - **`references`** (*list of dict*):
          A list of reference entries, where each entry is a dictionary with the following keys:
          - `source` (*any*): The source element of the reference.
          - `target` (*any*): The target element of the reference.
          - `relation` (*any*): The relationship between the source and target in the reference.

        Returns:
        ---------

        - **`dict`**:
          A dictionary containing the intersections and the calculated metrics:
          - `"intersections"`: A list of elements present in both `predicts` and `references`.
          - `"precision"`: The calculated precision score.
          - `"recall"`: The calculated recall score.
          - `"f_score"`: The calculated F-score with the specified \( β \).
    """
    intersection = calculate_intersection(predicts=predicts, references=references)
    precision = intersection / len(predicts) if len(predicts) != 0 else 0
    recall = intersection / len(references) if len(references) != 0 else 0
    beta_square = beta * beta
    f_score = (
        ((1 + beta_square) * precision * recall) / (beta_square * precision + recall)
        if (beta_square * precision + recall) != 0
        else 0
    )
    evaluations_dict = {
        "intersection": intersection,
        "precision": precision * 100,
        "recall": recall * 100,
        "f-score": f_score * 100,
        "predictions-len": len(predicts),
        "reference-len": len(references),
    }
    return evaluations_dict

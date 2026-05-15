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

    This function compares the predicted data with the reference data and determines
    the number of matching items. A match is identified when both the `source` and
    `target` fields are identical in a predicted-reference pair.

    The function now handles duplicate predictions by counting only unique matches.

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
    - **`intersection`** (*int*):
        The count of unique (source, target) pairs that appear in both predictions
        and references.
    """
    # Convert predictions to a set of unique (source, target) tuples
    predict_set = {(pred["source"], pred["target"]) for pred in predicts}
    # Convert references to a set of unique (source, target) tuples
    reference_set = {(ref["source"], ref["target"]) for ref in references}
    # Calculate intersection
    intersection = len(predict_set & reference_set)
    return intersection

def precision_score(predicts: List, references: List) -> float:
    r"""
    Calculate Precision:

    Precision is the proportion of predicted items that are correct. It is calculated as:
    $$\( P = \frac{|\text{intersection of predicts and references}|}{|\text{predicts}|} \)$$

    Parameters:
    ------------
    - **`predicts`** (*list of dict*): A list of predicted entries.
    - **`references`** (*list of dict*): A list of reference entries.

    Returns:
    ---------
    - **`precision`** (*float*): The calculated precision score.
    """
    intersection = calculate_intersection(predicts=predicts, references=references)
    return intersection / len(predicts) if len(predicts) != 0 else 0

def recall_score(predicts: List, references: List) -> float:
    r"""
    Calculate Recall:

    Recall is the proportion of reference items that are successfully predicted. It is calculated as:
    $$\( R = \frac{|\text{intersection of predicts and references}|}{|\text{references}|} \)$$

    Parameters:
    ------------
    - **`predicts`** (*list of dict*): A list of predicted entries.
    - **`references`** (*list of dict*): A list of reference entries.

    Returns:
    ---------
    - **`recall`** (*float*): The calculated recall score.
    """
    intersection = calculate_intersection(predicts=predicts, references=references)
    return intersection / len(references) if len(references) != 0 else 0

def f1_measurement(predicts: List, references: List, beta: int = 1) -> float:
    r"""
    Calculate F-Score:

    The F-score is a weighted harmonic mean of precision and recall, where \( β \) determines the balance between them. It is calculated as:
    $$\( F_β = \frac{(1 + β^2) \cdot P \cdot R}{(β^2 \cdot P) + R} \)$$

    Parameters:
    ------------
    - **`predicts`** (*list of dict*): A list of predicted entries.
    - **`references`** (*list of dict*): A list of reference entries.
    - **`beta`** (*int*): The weight of recall in the combined score (default is 1 for F1-score).

    Returns:
    ---------
    - **`f_score`** (*float*): The calculated F-score with the specified \( β \).
    """
    precision = precision_score(predicts=predicts, references=references)
    recall = recall_score(predicts=predicts, references=references)
    beta_square = beta * beta
    f_score = (
        ((1 + beta_square) * precision * recall) / (beta_square * precision + recall)
        if (beta_square * precision + recall) != 0
        else 0
    )
    return f_score

def evaluation_report(predicts: List, references: List, beta: int = 1) -> Dict:
    r"""
    :param predicts:
    :param references:
    :param beta:
    :return:
    """
    intersection = calculate_intersection(predicts=predicts, references=references)
    precision = precision_score(predicts=predicts, references=references)
    recall = recall_score(predicts=predicts, references=references)
    f_score = f1_measurement(predicts=predicts, references=references, beta=beta)

    evaluations_dict = {
        "intersection": intersection,
        "precision": precision * 100,
        "recall": recall * 100,
        "f-score": f_score * 100,
        "predictions-len": len(predicts),
        "reference-len": len(references),
    }
    return evaluations_dict

def hit_at_k(predicts: List, references: List, k: int = 1) -> float:
    """
    Compute Hit@K: fraction of references where the correct target appears in the
    top-K predicted targets for the same source.

    Assumes:
    - predicts: [{"source": str, "target": str, "score": float(optional)}]
    - references: [{"source": str, "target": str, ...}]
    """
    if k <= 0:
        return 0.0

    # Build per-source map of target->max(score)
    pred_map: Dict[str, Dict[str, float]] = {}
    for p in predicts:
        source = p.get("source")
        target = p.get("target")
        if source is None or target is None:
            continue
        score = float(p.get("score", 0.0))
        pred_map.setdefault(source, {})
        # keep the highest score if duplicates exist
        if target not in pred_map[source] or score > pred_map[source][target]:
            pred_map[source][target] = score

    if not references:
        return 0.0

    hits = 0
    n = len(references)
    for ref in references:
        source = ref.get("source")
        target = ref.get("target")
        if source is None or target is None:
            continue  # skip malformed reference; does not count as hit
        candidates = pred_map.get(source, {})
        if not candidates:
            # no predictions for this source → miss
            continue
        # rank targets for this source by descending score
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        topk = {item for item, _ in ranked[:k]}
        if target in topk:
            hits += 1

    return hits / n if n else 0.0

def mrr(predicts: List, references: List) -> float:
    """
    Mean Reciprocal Rank (MRR): average of reciprocal ranks for correct targets per source.

    For each reference (source, target), sort predicted candidates for that source by score
    descending, find the rank of the correct target (1-based). If missing, contributes 0.
    """
    # Build per-source map of target->max(score)
    pred_map: Dict[str, Dict[str, float]] = {}
    for p in predicts:
        source = p.get("source")
        target = p.get("target")
        if source is None or target is None:
            continue
        score = float(p.get("score", 0.0))
        pred_map.setdefault(source, {})
        if target not in pred_map[source] or score > pred_map[source][target]:
            pred_map[source][target] = score

    if not references:
        return 0.0

    rr_sum = 0.0
    n = len(references)
    for ref in references:
        source = ref.get("source")
        target = ref.get("target")
        if source is None or target is None:
            continue
        candidates = pred_map.get(source, {})
        if not candidates:
            continue
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        # find rank of target
        for i, (item, _score) in enumerate(ranked):
            if item == target:
                rr_sum += 1.0 / (i + 1)
                break
        # if not found → contributes 0
    return rr_sum / n if n else 0.0

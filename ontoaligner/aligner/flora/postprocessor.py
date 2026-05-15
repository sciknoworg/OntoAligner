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
from typing import List, Dict, Any, Tuple
from .fuzzy import bilateral_max_assign

def flora_bilateral_postprocess(same_as_scores, source_prefix) -> Dict[str, Dict[str, float]]:
    """
    Bilateral post-processing of KG alignments.

    Steps reproduced exactly:
      1) Build sameAsScores from flat predictions
      2) Run bilateral_max_assign(sameAsScores)
      3) Keep only top-1 match per source entity that starts with source_prefix

    No GT is used here.
    """
    max_assign = bilateral_max_assign(same_as_scores)
    filtered: Dict[str, Dict[str, float]] = {}
    for e1, matches in max_assign.items():
        if not e1.startswith(source_prefix):
            continue
        # take highest-score match only
        for e2 in matches:
            filtered.setdefault(e1, {})[e2] = matches[e2]
            break
    return filtered


def flora_1to1_postprocessor(same_as_scores) -> List[Tuple[str, str, float]]:
    """
    1-to-1 post-processing
    Correct behavior:
      - Gets score matrix
      - Apply bilateral_max_assign (NOT greedy matching)
      - Flatten to (source, target, score)
    """
    max_assign = bilateral_max_assign(same_as_scores)
    result: List[Tuple[str, str, float]] = []
    for s, targets in max_assign.items():
        for t, sc in targets.items():
            result.append((s, t, sc))
            break  # top-1 only
    return result





def flora_threshold_post_processor(predictions: List[Dict[str, Any]],
                                 prefix1: str,
                                 prefix2: str,
                                 threshold: float = 0.1) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """
    Threshold-style  post-processing.

    This version removes ALL GT dependency and uses only:
      - score thresholding
      - prefix filtering
      - per-source best selection
      - simple property symmetry heuristic
    """

    inst_pred: Dict[str, Dict[str, float]] = {}
    rel_pred: Dict[str, Dict[str, float]] = {}

    # ---- split predictions WITHOUT GT ----
    for pred in predictions:
        s = pred["source"]
        t = pred["target"]
        sc = float(pred["score"])
        ptype = pred.get("type", "instance")

        if sc < threshold:
            continue

        if ptype == "predicate":
            rel_pred.setdefault(s, {})[t] = sc
        else:
            inst_pred.setdefault(s, {})[t] = sc  # unified handling

    # ---- instances (top-1 per source, prefix-aware) ----
    inst_align: Dict[str, Dict[str, float]] = {}

    for s, targets in inst_pred.items():
        candidates = [
            (t, sc) for t, sc in targets.items()
            if t.startswith(prefix2)
        ]
        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        t, sc = candidates[0]

        inst_align.setdefault(t, {})[s] = sc

    # ---- properties (simple symmetric + top-1) ----
    rel_align: Dict[str, Dict[str, float]] = {}

    for s, targets in rel_pred.items():
        candidates = sorted(targets.items(), key=lambda x: x[1], reverse=True)

        for t, sc in candidates:
            if t.startswith(prefix2):
                rel_align.setdefault(t, {})[s] = sc
                break

    # ---- classes (no GT → treated same as instances safely) ----
    cls_align: Dict[str, Dict[str, float]] = {}

    for s, targets in inst_pred.items():
        candidates = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        if not candidates:
            continue

        t, sc = candidates[0]
        if t.startswith(prefix2):
            cls_align.setdefault(t, {})[s] = sc

    return inst_align, cls_align, rel_align

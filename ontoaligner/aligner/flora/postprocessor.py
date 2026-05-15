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

def __map_to_alignment_triples(filtered_items: Dict[str, Dict[str, float]]) -> List[Dict[str, str|float]]:
    alignments = []
    for source, target_values in filtered_items.items():
        for target, score in target_values.items():
            alignments.append({
                "source": source.replace("<", "").replace(">", ""),
                "target": target.replace("<", "").replace(">", ""),
                "score": score,
            })
    return alignments

def __map_to_alignment_triples_swapped(filtered_items: Dict[str, Dict[str, float]]) -> List[Dict[str, str | float]]:
    alignments = []
    for target, source_values in filtered_items.items():
        for source, score in source_values.items():
            alignments.append({
                "source": source.replace("<", "").replace(">", ""),
                "target": target.replace("<", "").replace(">", ""),
                "score": score,
            })
    return alignments

def flora_bilateral_postprocessor(same_as_scores, source_prefix) -> List[Dict[str, str|float]]:
    """
    Bilateral post-processing of KG alignments.

    Steps reproduced exactly:
      1) Build sameAsScores from flat predictions
      2) Run bilateral_max_assign(sameAsScores)
      3) Keep only top-1 match per source entity that starts with source_prefix
    """
    max_assign = bilateral_max_assign(same_as_scores)
    filtered: Dict[str, Dict[str, float]] = {}
    for e1, matches in max_assign.items():
        if not e1.startswith('<'+source_prefix):
            continue
        # take highest-score match only
        for e2 in matches:
            filtered.setdefault(e1, {})[e2] = matches[e2]
            break
    return __map_to_alignment_triples(filtered)


def flora_1to1_postprocessor(same_as_scores) -> List[Dict[str, str|float]]:
    """
    1-to-1 post-processing
    Correct behavior:
      - Gets score matrix
      - Apply bilateral_max_assign (NOT greedy matching)
      - Flatten to (source, target, score)
    """
    max_assign = bilateral_max_assign(same_as_scores)
    return __map_to_alignment_triples(max_assign)


def flora_threshold_postprocessor(predictions: List[Dict[str, Any]],
                                 prefix1: str,
                                 prefix2: str,
                                 threshold: float = 0.1) -> Tuple:
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
        source = pred["source"]
        target = pred["target"]
        score = float(pred["score"])
        ptype = pred.get("type", "instance")

        if score < threshold:
            continue

        if ptype == "predicate":
            rel_pred.setdefault(source, {})[target] = score
        else:
            inst_pred.setdefault(source, {})[target] = score  # unified handling

    # ---- instances (top-1 per source, prefix-aware) ----
    inst_align: Dict[str, Dict[str, float]] = {}

    for s, targets in inst_pred.items():
        candidates = [
            (t, sc) for t, sc in targets.items()
            if t.startswith('<' + prefix2)
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
            if t.startswith('<' + prefix2):
                rel_align.setdefault(t, {})[s] = sc
                break

    # ---- classes (no GT → treated same as instances safely) ----
    cls_align: Dict[str, Dict[str, float]] = {}

    for s, targets in inst_pred.items():
        candidates = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        if not candidates:
            continue

        t, sc = candidates[0]
        if t.startswith('<' + prefix2):
            cls_align.setdefault(t, {})[s] = sc

    instance_alignments = __map_to_alignment_triples_swapped(inst_align)
    class_alignments = __map_to_alignment_triples_swapped(cls_align)
    predicate_alignments = __map_to_alignment_triples_swapped(rel_align)

    return instance_alignments, class_alignments, predicate_alignments

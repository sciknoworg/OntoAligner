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
Core alignment algorithms for the FLORA (Fuzzy Logic KG Alignment) system.

This module implements the fuzzy-logic inference rules and iterative procedures
that form the heart of the FLORA algorithm as described in:

    Peng, Yiwen, Bonald, Thomas, and Suchanek, Fabian.
    "FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic."
    ISWC 2025. https://suchanek.name/work/publications/iswc-2025.pdf

Key components:

- **Predicate subsumption** – :func:`initialize_predicate_subsumption`,
  :func:`update_predicate_subsumption`, :func:`compute_quasi_eqrel`.
- **Functionality computation** – :func:`compute_functionalities`,
  :func:`compute_functionalities_for_predicates`.
- **Implication functions** – :func:`update_score_min` (Gödel min-norm, Eq. 1),
  :func:`update_score_additive_min` (additive normalised, Eq. 2).
- **Bootstrapping** – :func:`bootstrap_algo`, :func:`first_iteration`.
- **Relation mapping** – :func:`map_subrelations`.
- **Bilateral max assignment** – :func:`bilateral_max_assign` (Eq. 3).
- **Parallel entity matching** – :func:`_match_entities_by_rules` (Eq. 2,
  multi-process worker).
"""

from itertools import combinations
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Set, Tuple
from scipy import stats
import numpy as np
import pandas as pd
from . import literals


# Constants for accessing the components of a triple
SUBJ = 0
PRED = 1
OBJ = 2


#################################################################
#               Predicates and Functionalities                  #
#################################################################
def initialize_predicate_subsumption(
    predicates1: Set[Any],
    predicates2: Set[Any],
    pred2super_pred12: Optional[Dict[Any, Dict[Any, float]]] = None,
    pred2super_pred21: Optional[Dict[Any, Dict[Any, float]]] = None,
    relinit: float = 0.1,
) -> Dict[Any, Dict[Any, float]]:
    """Initialize predicate subsumption scores between two knowledge bases.

    Sets identical relations to 1.0, and initializes others with provided scores
    or a default initial value.

    Args:
        predicates1: Set of predicates in KB1.
        predicates2: Set of predicates in KB2.
        pred2super_pred12: Optional subsumption scores from KB1 predicates to KB2.
        pred2super_pred21: Optional subsumption scores from KB2 predicates to KB1.
        relinit: Initial score for non-identical relations. Defaults to 0.1.

    Returns:
        Nested dictionary of pairwise subsumption scores across KGs in both directions.
    """
    if pred2super_pred12 is None:
        pred2super_pred12 = {}
    if pred2super_pred21 is None:
        pred2super_pred21 = {}

    result = {}
    for pred1 in predicates1:
        if pred1 not in result:
            result[pred1] = {}
        for pred2 in predicates2:
            if pred2 not in result:
                result[pred2] = {}
            if pred1 == pred2:
                result[pred1][pred2] = 1.0
            else:
                score1 = max(
                    pred2super_pred12.get(pred1, {}).get(pred2, relinit),
                    pred2super_pred21.get(pred1, {}).get(pred2, relinit),
                )
                score2 = max(
                    pred2super_pred21.get(pred2, {}).get(pred1, relinit),
                    pred2super_pred12.get(pred2, {}).get(pred1, relinit),
                )
                result[pred1][pred2] = score1
                result[pred2][pred1] = score2
    return result


def update_predicate_subsumption(
    pred2super_pred12: Dict[Any, Dict[Any, float]],
    pred2super_pred21: Dict[Any, Dict[Any, float]],
    previous_predicate2super_predicate: Optional[Dict[Any, Dict[Any, float]]] = None,
) -> Dict[Any, Dict[Any, float]]:
    """Update predicate subsumption scores bidirectionally.

    Updates subsumption scores from KB1→KB2 and KB2→KB1, maintaining
    monotonicity of relation subsumption. Returns a new dictionary rather
    than modifying in-place.

    Args:
        pred2super_pred12: Current subsumption scores from KB1 to KB2 predicates.
        pred2super_pred21: Current subsumption scores from KB2 to KB1 predicates.
        previous_predicate2super_predicate: Previous subsumption scores to be updated.
            If None, an empty dictionary is created.

    Returns:
        Updated predicate subsumption scores dictionary.
    """
    result = previous_predicate2super_predicate if previous_predicate2super_predicate is not None else {}

    for pred1 in pred2super_pred12:
        if result.get(pred1) is None:
            result[pred1] = {}
        for pred2 in pred2super_pred12[pred1]:
            # Make relation subsumption monotonic
            result[pred1][pred2] = max(
                result[pred1].get(pred2, 0),
                pred2super_pred12[pred1][pred2],
            )
    for pred2 in pred2super_pred21:
        if result.get(pred2) is None:
            result[pred2] = {}
        for pred1 in pred2super_pred21[pred2]:
            # Make relation subsumption monotonic
            result[pred2][pred1] = max(
                result[pred2].get(pred1, 0),
                pred2super_pred21[pred2][pred1],
            )
    return result


def compute_functionalities(kb: Any, gram: Optional[List[int]] = None) -> Dict[Any, float]:
    """Compute functionality scores for predicates in a knowledge base.

    Functionality is measured as the ratio of unique subjects per predicate,
    considering n-gram combinations for higher-order relationships.

    Args:
        kb: The input knowledge base graph object.
        gram: List of integers indicating n-gram sizes to consider. Defaults to [].

    Returns:
        Dictionary mapping predicates to their functionality scores.
    """
    if gram is None:
        gram = []

    predicate2num_facts = {}
    predicate2subjects = {}

    for subject in kb.subjects():
        facts = list(kb.triplesWithSubject(subject))
        for n in gram:
            if n == 1:
                for fact in facts:
                    predicate = fact[PRED]
                    if predicate not in predicate2num_facts:
                        predicate2num_facts[predicate] = 0
                        predicate2subjects[predicate] = set()
                    predicate2num_facts[predicate] += 1
                    predicate2subjects[predicate].add(fact[SUBJ])
                continue

            # n > 1: compute n-gram combinations
            count = 0
            for evidences in combinations(facts, n):
                count += 1
                predicate = tuple(sorted([literals.invert(fact[PRED]) for fact in evidences]))
                subjects = tuple(sorted([fact[OBJ] for fact in evidences]))
                if predicate not in predicate2num_facts:
                    predicate2num_facts[predicate] = 0
                    predicate2subjects[predicate] = set()
                predicate2num_facts[predicate] += 1
                predicate2subjects[predicate].add(subjects)
                if count > 100000:  # Avoid memory overflow
                    break

    return {
        predicate: len(predicate2subjects[predicate]) / predicate2num_facts[predicate]
        for predicate in predicate2num_facts
    }


#################################################################
#                 Implication Functions                         #
#################################################################

def update_score_min(
    mapping: Dict[Any, Dict[Any, float]],
    key1: Any,
    key2: Any,
    *body: float,
) -> Dict[Any, Dict[Any, float]]:
    """Update score using minimum operator (Gödel logic).

    Updates mapping[key1][key2] so that the rule body=>mapping[key1][key2] holds
    using the minimum operator, as shown in equation (1) in the FLORA paper.
    Returns the updated mapping dictionary.

    Args:
        mapping: Nested dictionary to be updated with entity alignment scores.
        key1: The entity from KB1.
        key2: The entity from KB2.
        *body: Values in the body of the rule.

    Returns:
        Updated mapping dictionary with new alignment scores.
    """
    current_score = 0.0
    if key1 in mapping and key2 in mapping.get(key1, {}):
        current_score = mapping[key1][key2]

    temp = max(current_score, min(min(body), 1.0))

    if temp > 0:
        if key1 not in mapping:
            mapping[key1] = {}
        mapping[key1][key2] = temp

    return mapping


def update_score_additive_min(
    mapping: Dict[Any, Dict[Any, float]],
    key1: Any,
    key2: Any,
    factor: float,
    *body: float,
) -> Dict[Any, Dict[Any, float]]:
    """Update score using additive minimum operator.

    Updates mapping[key1][key2] by adding the rule value. Used for subrelation
    rules, as shown in equation (2) in the FLORA paper. Returns the updated mapping.

    Args:
        mapping: Nested dictionary to be updated with subrelation scores.
        key1: The predicate from KB1.
        key2: The predicate from KB2.
        factor: Normalization factor (already multiplied by benefit-of-doubt parameter).
        *body: Values in the body of the rule.

    Returns:
        Updated mapping dictionary with new subrelation scores.
    """
    current_score = 0.0
    if key1 in mapping and key2 in mapping.get(key1, {}):
        current_score = mapping[key1][key2]

    value = current_score + min(body) * factor
    if value > 0:
        if key1 not in mapping:
            mapping[key1] = {}
        mapping[key1][key2] = max(min(value, 1.0), 0)

    return mapping


def update_max_score_min(
    mapping: Dict[Any, Tuple[Tuple, float]],
    pred: Any,
    fact: Tuple,
    *body: float,
) -> Dict[Any, Tuple[Tuple, float]]:
    """Update mapping with the maximum aligned scoring fact for each predicate.

    Used in subrelation rules to track the best matching facts.
    Returns the updated mapping dictionary.

    Args:
        mapping: Dictionary to be updated with (fact, score) tuples per predicate.
        pred: The predicate from KB2.
        fact: The fact tuple (subject, predicate, object).
        *body: Values in the body of the rule.

    Returns:
        Updated mapping dictionary with maximum scoring facts.
    """
    score = min(body)
    if pred not in mapping:
        mapping[pred] = (fact, score)
    else:
        if score > mapping[pred][1]:
            mapping[pred] = (fact, score)

    return mapping


#################################################################
#                      Procedure                                #
#################################################################
def first_iteration(
    kb_src: Any,
    kb_dst: Any,
    pred2super_pred: Dict[Any, Dict[Any, float]],
    functionalities: Dict[Any, float],
    queue: mp.Queue,
    ent_match_tuple_queue: mp.Queue,
    ent_max_assign: Dict[Any, Dict[Any, float]],
) -> None:
    """First iteration used for bootstrapping the algorithm using initial literal alignments.

    This is the main worker function for the bootstrapping phase, run in parallel
    across multiple processes. It processes entities from the queue and computes
    initial entity alignment scores based on predicate subsumption and functionality.

    Results are put into ent_match_tuple_queue for collection by the parent process.

    Args:
        kb_src: The source knowledge base.
        kb_dst: The target knowledge base.
        pred2super_pred: Nested dictionary of pairwise subsumption scores.
        functionalities: Dictionary mapping predicates to their functionality scores.
        queue: Multiprocessing queue containing entities to be aligned.
        ent_match_tuple_queue: Queue to store resulting entity alignment scores.
        ent_max_assign: Bilateral max assignment from initial literal alignments.
    """
    ent_match_scores = dict()
    while not queue.empty():
        try:
            subj_kb1 = queue.get_nowait()
        except Exception:
            break

        for fact1 in kb_src.triplesWithSubject(subj_kb1):
            # We don't match literals
            if literals.is_literal(fact1[OBJ]):
                continue
            # Continue if the subject has not been matched
            if fact1[SUBJ] not in ent_max_assign:
                continue
            for subj_kb2 in ent_max_assign[fact1[SUBJ]]:
                if subj_kb2 not in kb_dst.index:
                    continue
                for fact2 in kb_dst.triplesWithSubject(subj_kb2, pred2super_pred[fact1[PRED]]):
                    # We don't match literals
                    if literals.is_literal(fact2[OBJ]):
                        continue
                    # Update
                    ent_match_scores = update_score_min(
                        # Objects are the same, ...
                        ent_match_scores, fact1[OBJ], fact2[OBJ],
                        # ... if the subjects are the same, ...
                        ent_max_assign[fact1[SUBJ]][fact2[SUBJ]],
                        # ... and the predicate is locally functional, ...
                        kb_src.localFunctionality(fact1[SUBJ], fact1[PRED]),
                        kb_dst.localFunctionality(fact2[SUBJ], fact2[PRED]),
                        # ... and the predicate is globally functional,
                        functionalities[fact1[PRED]], functionalities[fact2[PRED]],
                        # ... and the target predicate is subsumed.
                        max(pred2super_pred[fact1[PRED]][fact2[PRED]],
                            pred2super_pred[fact2[PRED]][fact1[PRED]])
                    )
    # Update the queue
    ent_match_tuple_queue.put(ent_match_scores)
    exit(1)


def bootstrap_algo(
    kb_src: Any,
    kb_dst: Any,
    same_as_score: Dict[Any, Dict[Any, float]],
    pred2super_pred: Dict[Any, Dict[Any, float]],
    functionalities: Dict[Any, float],
) -> Dict[Any, Dict[Any, float]]:
    """Bootstrap the algorithm using initial literal alignments.

    This function runs the first iteration in parallel to initialize entity
    alignment scores based on the initial literal similarity alignments.

    Args:
        kb_src: The source knowledge base.
        kb_dst: The target knowledge base.
        same_as_score: Nested dictionary of entity alignment scores
            (includes initial literal alignments).
        pred2super_pred: Nested dictionary of pairwise subsumption scores.
        functionalities: Dictionary mapping predicates to their functionality scores.

    Returns:
        Updated same_as_score dictionary with bootstrapped entity alignments.
    """
    ent_max_assign = bilateral_max_assign(same_as_score)
    mgr = mp.Manager()
    subjs_kb1 = kb_src.subjects()
    ent_queue = mgr.Queue(len(subjs_kb1))
    for subj_kb1 in subjs_kb1:
        ent_queue.put(subj_kb1)
    tasks = []
    num_workers = 90
    ent_match_tuple_queue = mgr.Queue()
    for _ in range(num_workers):
        task = mp.Process(
            target=first_iteration,
            args=(
                kb_src, kb_dst,
                pred2super_pred,
                functionalities,
                ent_queue,
                ent_match_tuple_queue,
                ent_max_assign,
            ),
        )
        task.start()
        tasks.append(task)
    for task in tasks:
        task.join()
    while not ent_match_tuple_queue.empty():
        ent_match_score_dict = ent_match_tuple_queue.get()
        # Update same_as_score using max aggregation
        for subj1 in ent_match_score_dict:
            if subj1 not in same_as_score:
                same_as_score[subj1] = {}
            for subj2 in ent_match_score_dict[subj1]:
                if ent_match_score_dict[subj1][subj2] > same_as_score[subj1].get(subj2, 0):
                    same_as_score[subj1][subj2] = ent_match_score_dict[subj1][subj2]
    return same_as_score


def map_subrelations(
    alpha: float,
    kb_src: Any,
    kb_dst: Any,
    ent_max_assign: Dict[Any, Dict[Any, float]],
    previous_predicate2super_predicate: Dict[Any, Dict[Any, float]],
) -> Dict[Any, Dict[Any, float]]:
    """Map subrelations in both directions using current entity alignments.

    Updates predicate subsumption scores based on aligned entity pairs,
    computing which predicates in one KB correspond to predicates in the other.

    Args:
        alpha: Benefit-of-doubt parameter for subrelation mapping.
        kb_src: The source knowledge base.
        kb_dst: The target knowledge base.
        ent_max_assign: Bilateral max assignment from current entity alignments.
        previous_predicate2super_predicate: Previous subsumption scores to be updated.

    Returns:
        Updated predicate subsumption scores dictionary.
    """
    # Match predicates - Direction: kb1 -> kb2
    pred2super_pred1 = {}
    for fact1 in kb_src:
        if fact1[SUBJ] not in ent_max_assign:
            continue
        # For each fact1, find the best matching fact2 for each relation r2
        # So that for the given relation pair (r1, r2), there is one most matched fact2
        rel2max_fact = {}  # {rel2: (fact2, score)}
        for subject2 in ent_max_assign[fact1[SUBJ]]:
            if subject2 not in kb_dst.index:
                continue
            for fact2 in kb_dst.triplesWithSubject(subject2):
                # Skip if objects are not aligned
                if fact1[OBJ] not in ent_max_assign:
                    continue
                if fact2[OBJ] not in ent_max_assign.get(fact1[OBJ], {}):
                    continue
                score_object = ent_max_assign[fact1[OBJ]][fact2[OBJ]]
                update_max_score_min(
                    rel2max_fact, fact2[PRED], fact2,
                    ent_max_assign[fact1[SUBJ]][fact2[SUBJ]], score_object
                )
        # Update predicate subsumption
        for pred2, (fact2, score) in rel2max_fact.items():
            pred2super_pred1 = update_score_additive_min(
                pred2super_pred1, fact1[PRED], fact2[PRED],
                alpha / kb_src.numFactsWithPredicate(fact1[PRED]), score
            )

    # Direction: kb2 -> kb1
    pred2super_pred2 = {}
    for fact2 in kb_dst:
        if fact2[SUBJ] not in ent_max_assign:
            continue
        rel1max_fact = {}  # {rel1: (fact1, score)}
        for subject1 in ent_max_assign[fact2[SUBJ]]:
            if subject1 not in kb_src.index:
                continue
            for fact1 in kb_src.triplesWithSubject(subject1):
                # Skip if objects are not aligned
                if fact2[OBJ] not in ent_max_assign:
                    continue
                if fact1[OBJ] not in ent_max_assign.get(fact2[OBJ], {}):
                    continue
                score_object = ent_max_assign[fact2[OBJ]][fact1[OBJ]]
                update_max_score_min(
                    rel1max_fact, fact1[PRED], fact1,
                    ent_max_assign[fact2[SUBJ]][fact1[SUBJ]], score_object
                )
        # Update predicate subsumption
        for pred1, (fact1, score) in rel1max_fact.items():
            pred2super_pred2 = update_score_additive_min(
                pred2super_pred2, fact2[PRED], fact1[PRED],
                alpha / kb_dst.numFactsWithPredicate(fact2[PRED]), score
            )
    # Complete the subrelation mapping
    previous_predicate2super_predicate = update_predicate_subsumption(pred2super_pred1, pred2super_pred2, previous_predicate2super_predicate)
    return previous_predicate2super_predicate


def compute_quasi_eqrel(
    kb_src: Any,
    kb_dst: Any,
    pred2super_pred: Dict[Any, Dict[Any, float]],
) -> Dict[Any, Dict[Any, float]]:
    """Compute quasi equivalence relations between two KGs' predicates.

    The quasi equivalence is represented as r≅r' in the FLORA paper.

    Args:
        kb_src: The source knowledge base.
        kb_dst: The target knowledge base.
        pred2super_pred: Nested dictionary of pairwise subsumption scores.

    Returns:
        Nested dictionary of quasi equivalence relations.
    """
    quasi_eqrel = {}  # from kb1 to kb2
    for pred1 in pred2super_pred:
        for pred2 in pred2super_pred[pred1]:
            value = max(
                pred2super_pred[pred1][pred2],
                pred2super_pred.get(pred2, {}).get(pred1, 0)
            )
            if pred1 in kb_src.predicates():
                if pred2 in kb_dst.predicates():
                    if pred1 not in quasi_eqrel:
                        quasi_eqrel[pred1] = {}
                    quasi_eqrel[pred1][pred2] = value
            elif pred2 in kb_src.predicates() and pred2 not in quasi_eqrel:
                quasi_eqrel[pred2] = {}
                quasi_eqrel[pred2][pred1] = value
    return quasi_eqrel


def bilateral_max_assign(same_as_score: Dict[Any, Dict[Any, float]]) -> Dict[Any, Dict[Any, float]]:
    """Compute bilateral max assignment from similarity scores.

    Computes the bilateral max assignment, as described in equation (3) of the
    FLORA paper.

    Args:
        same_as_score: Nested dictionary of entity alignment scores.

    Returns:
        The bilateral max assignment of entities.
    """
    # get max match for kb1
    max_match_e = {}
    for e, e_prime_scores in same_as_score.items():
        if not e_prime_scores:
            continue
        max_score = max(e_prime_scores.values())
        max_targets_e = []
        for e_prime, score in e_prime_scores.items():
            if score == max_score:
                max_targets_e.append(e_prime)
        if max_targets_e:
            max_match_e[e] = {'score': max_score, 'targets': set(max_targets_e)}

    # get same_as_score inverse: kb2 -> kb1
    e_prime_to_e_scores = {}
    for e, e_prime_scores in same_as_score.items():
        for e_prime, score in e_prime_scores.items():
            if e_prime not in e_prime_to_e_scores:
                e_prime_to_e_scores[e_prime] = {}
            e_prime_to_e_scores[e_prime][e] = score

    # get max match for kb2
    max_match_e_prime = {}
    for e_prime, e_scores in e_prime_to_e_scores.items():
        max_score_e_prime = max(e_scores.values())
        max_targets_e_prime = []
        for e, score in e_scores.items():
            if score == max_score_e_prime:
                max_targets_e_prime.append(e)
        if max_targets_e_prime:
            max_match_e_prime[e_prime] = {'score': max_score_e_prime, 'targets': set(max_targets_e_prime)}

    # bilateral max assignment
    res_max_assign = {}
    for e, match_data in max_match_e.items():
        max_score = match_data['score']
        max_targets = match_data['targets']

        for e_prime in max_targets:
            if (e_prime in max_match_e_prime and
                max_match_e_prime[e_prime]['score'] == max_score and
                e in max_match_e_prime[e_prime]['targets']):
                # write (e, e')
                if e not in res_max_assign:
                    res_max_assign[e] = {}
                res_max_assign[e][e_prime] = max_score
                # write (e', e)
                if e_prime not in res_max_assign:
                    res_max_assign[e_prime] = {}
                res_max_assign[e_prime][e] = max_score
    return res_max_assign



def _match_entities_by_rules(
    kb_src: Any,
    kb_dst: Any,
    quasi_eqvrel: Dict[Any, Dict[Any, float]],
    queue: mp.Queue,
    ent_match_tuple_queue: mp.Queue,
    same_as_score: Dict[Any, Dict[Any, float]],
    functionalities: Dict[Any, float],
    params: Dict[str, Any],
) -> None:
    """Match entities in parallel using fuzzy logic rules.

    Corresponds to equation (2) in the FLORA paper. The function consists of two parts:
    candidate search and entity alignment.

    Args:
        kb_src: The source knowledge base.
        kb_dst: The target knowledge base.
        quasi_eqvrel: Nested dictionary of quasi equivalence relations.
        queue: Multiprocessing queue containing the entities to be aligned.
        ent_match_tuple_queue: Queue to store the resulting entity alignment scores.
        same_as_score: Nested dictionary of all entity alignment scores (includes initial literal alignments).
        functionalities: Dictionary mapping predicates to their functionalities.
        params: Dictionary of parameters, including 'gramN' (the maximum n-gram size to consider).
    """
    ent_match_scores = dict()
    ent_max_assign = bilateral_max_assign(same_as_score)
    while not queue.empty():
        try:
            subj_kb1 = queue.get_nowait()
        except Exception:
            break

        # We don't need to match literals
        if literals.is_literal(subj_kb1):
            continue

        # Skip if the entity is already matched
        if subj_kb1 in ent_max_assign and \
            round(max(ent_max_assign.get(subj_kb1, {None: 0}).values()), 1) >= 1.0:
            continue

        # Search Algorithm
        kb1_facts_ordered = []
        for fact1 in kb_src.triplesWithSubject(subj_kb1):
            if max(ent_max_assign.get(fact1[OBJ], {None: 0}).values()) <= 0:
                continue
            if max(quasi_eqvrel.get(fact1[PRED], {None: 0}).values()) <= 0:
                continue
            kb1_facts_ordered.append((fact1[OBJ], literals.invert(fact1[PRED]), subj_kb1))
        # search order: the most informative facts first
        kb1_facts_ordered.sort(
            reverse=True,
            key=lambda x: min(
                max(ent_max_assign[x[0]].values()),
                max(quasi_eqvrel[x[1]].values())
            )
        )
        subj2evi1 = dict()  # a dict of list of ordered evidences
        subj2evi2 = dict()  # {subj2: [ev2, ...]}
        for fact_kb1 in kb1_facts_ordered[:params['gramN']]:
            pred_kb1, obj_kb1 = fact_kb1[PRED], fact_kb1[0]
            # find the corresponding facts in kb2
            tmp_subj2_evi2 = dict()
            subj2_max_subrel_score = dict()
            for obj_kb2 in ent_max_assign[obj_kb1]:
                if obj_kb2 not in kb_dst.index:
                    continue
                aligned_evi2 = []
                max_subrel_score = 0
                for evi2_ in kb_dst.triplesWithSubject(obj_kb2):
                    if literals.is_literal(evi2_[OBJ]):
                        continue
                    subrel_score = quasi_eqvrel[pred_kb1].get(evi2_[PRED], 0)
                    if subrel_score <= 0:
                        continue
                    if subrel_score > max_subrel_score:
                        max_subrel_score = subrel_score
                        aligned_evi2 = [evi2_]
                    if subrel_score == max_subrel_score:
                        aligned_evi2.append(evi2_)
                if len(aligned_evi2) == 0:
                    continue
                for evi2 in aligned_evi2:
                    subj2_ = evi2[OBJ]
                    if subj2_ not in subj2_max_subrel_score:
                        subj2_max_subrel_score[subj2_] = 0
                    # certain evi1 (subj1, obj1, pred1),
                    # one subj2 has just one corresponding evidence2 at most
                    if quasi_eqvrel[pred_kb1][evi2[PRED]] > subj2_max_subrel_score[subj2_]:
                        subj2_max_subrel_score[subj2_] = quasi_eqvrel[pred_kb1][evi2[PRED]]
                        tmp_subj2_evi2[subj2_] = evi2
            # update subj2evi1 for exact evidence1 == fact_kb1
            for subj2, single_evi2 in tmp_subj2_evi2.items():
                if subj2 not in subj2evi1:
                    subj2evi1[subj2] = [fact_kb1]
                    subj2evi2[subj2] = [single_evi2]
                    continue
                # Reduce duplicates
                if single_evi2 in subj2evi2[subj2]:
                    index_evi2 = subj2evi2[subj2].index(single_evi2)
                    # compare scores
                    score1 = min(
                        ent_max_assign[subj2evi1[subj2][index_evi2][0]][single_evi2[SUBJ]],
                        quasi_eqvrel[subj2evi1[subj2][index_evi2][PRED]][single_evi2[PRED]]
                    )
                    score2 = min(
                        ent_max_assign[fact_kb1[0]][single_evi2[SUBJ]],
                        quasi_eqvrel[fact_kb1[PRED]][single_evi2[PRED]]
                    )
                    if score2 > score1:
                        subj2evi1[subj2][index_evi2] = fact_kb1
                        subj2evi2[subj2][index_evi2] = single_evi2
                    continue
                subj2evi1[subj2].append(fact_kb1)
                subj2evi2[subj2].append(single_evi2)

        # Selection Algorithm
        # select the entities with the most evidences
        subj2_count = dict()
        max_count = 0
        for subj2 in subj2evi2:
            if subj2 in ent_max_assign and \
                round(max(ent_max_assign.get(subj2, {None: 0}).values()), 1) >= 1.0:
                continue
            cur_count = len(set(subj2evi2[subj2]))
            if cur_count > max_count:
                subj2_count = dict()
                max_count = cur_count
                subj2_count[subj2] = len(set(subj2evi2[subj2]))
            elif cur_count == max_count:
                subj2_count[subj2] = len(set(subj2evi2[subj2]))
        if len(subj2_count) == 0:
            continue

        # Alignment Algorithm
        # Apply rules in order to update the scores
        gram_n = min(20, max_count)
        for subj_kb2 in subj2_count:
            assert len(subj2evi1[subj_kb2]) == len(subj2evi2[subj_kb2])

            # Re-order the list
            index_sorted = sorted(
                range(len(subj2evi1[subj_kb2])),
                reverse=True,
                key=lambda i: min(
                    ent_max_assign[subj2evi1[subj_kb2][i][0]][subj2evi2[subj_kb2][i][SUBJ]],
                    quasi_eqvrel[subj2evi1[subj_kb2][i][PRED]][subj2evi2[subj_kb2][i][PRED]]
                )
            )
            subj2evi1[subj_kb2] = [subj2evi1[subj_kb2][i] for i in index_sorted]
            subj2evi2[subj_kb2] = [subj2evi2[subj_kb2][i] for i in index_sorted]

            # find the common patterns
            visited_facts = set()
            ev1s, ev2s = subj2evi1[subj_kb2], subj2evi2[subj_kb2]
            # Try all possible sets
            for n in range(1, gram_n + 1):
                ev1, ev2 = ev1s[:n], ev2s[:n]
                if (tuple(ev1), tuple(ev2)) in visited_facts:
                    continue
                visited_facts.add((tuple(ev1), tuple(ev2)))
                obj1_combo, pred1_combo, subj1_combo = zip(*ev1)
                obj2_combo, pred2_combo, subj2_combo = zip(*ev2)
                # check if subjects itself are the same
                assert len(set(subj1_combo)) == 1
                assert len(set(subj2_combo)) == 1
                # check same pattern
                if not (pd.factorize(np.array(obj1_combo))[0]
                        == pd.factorize(np.array(obj2_combo))[0]).all():
                    continue
                localfunc1 = kb_src.localFunctionality(obj1_combo, pred1_combo)
                localfunc2 = kb_dst.localFunctionality(obj2_combo, pred2_combo)
                pred1_sort = tuple(sorted(list(pred1_combo)))
                pred2_sort = tuple(sorted(list(pred2_combo)))
                globalfunc1 = functionalities.get(pred1_sort, 1.0)
                globalfunc2 = functionalities.get(pred2_sort, 1.0)

                obj_eq = float(stats.hmean([ent_max_assign[obj1_combo[i]][obj2_combo[i]] for i in range(len(obj1_combo))]))
                pred_eq = float(stats.hmean([quasi_eqvrel[pred1_combo[i]][pred2_combo[i]] for i in range(len(pred1_combo))]))
                # update
                if n == 1:
                    ent_match_scores = update_score_min(
                        ent_match_scores, subj1_combo[0], subj2_combo[0],
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        functionalities[pred1_combo[0]], functionalities[pred2_combo[0]]
                    )
                else:
                    ent_match_scores = update_score_min(
                        ent_match_scores, subj1_combo[0], subj2_combo[0],
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        globalfunc1, globalfunc2,
                    )
    # Update the queue
    ent_match_tuple_queue.put(ent_match_scores)

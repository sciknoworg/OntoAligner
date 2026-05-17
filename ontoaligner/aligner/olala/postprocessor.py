"""
This script defines OLaLa postprocessing utilities.

The postprocessor merges LLM and high-precision outputs, applies bad-host
filtering, extracts one-to-one correspondences, and applies confidence filtering.
"""
from typing import Any, Dict, List
from urllib.parse import urlparse
import numpy as np

def get_host_of_uri(uri: str) -> str:
    """
    Extracts the host of a URI.

    Parameters:
        uri (str): The URI.

    Returns:
        str: The host of the URI.
    """
    if uri is None or uri == "":
        return ""

    try:
        host = urlparse(str(uri)).hostname
        if host is None:
            return ""
        return host
    except ValueError:
        return ""

def keep_correspondence_by_host(
    source_iri: str,
    target_iri: str,
    expected_source_host: str,
    expected_target_host: str,
    strict: bool = False,
) -> bool:
    """
    Checks whether a correspondence should be kept by host filtering.

    Parameters:
        source_iri (str): The source entity IRI.
        target_iri (str): The target entity IRI.
        expected_source_host (str): The expected source ontology host.
        expected_target_host (str): The expected target ontology host.
        strict (bool): Whether unknown hosts should be filtered.

    Returns:
        bool: True if the correspondence should be kept.
    """
    if not expected_source_host or not expected_target_host:
        return True

    source_host = get_host_of_uri(source_iri)
    target_host = get_host_of_uri(target_iri)

    if not source_host or not target_host:
        return not strict

    return (
        source_host == expected_source_host
        and target_host == expected_target_host
    )

def flatten_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts grouped or flat predictions into flat correspondences.

    Parameters:
        predictions (List[Dict[str, Any]]): The predictions.

    Returns:
        List[Dict[str, Any]]: The flattened correspondences.
    """
    flattened = []

    for prediction in predictions:
        if "target-cands" in prediction:
            source = prediction["source"]
            target_cands = prediction.get("target-cands", [])
            score_cands = prediction.get("score-cands", [])

            for target, score in zip(target_cands, score_cands):
                flattened.append({
                    "source": source,
                    "target": target,
                    "score": float(score),
                })
        else:
            flattened.append({
                "source": prediction["source"],
                "target": prediction["target"],
                "score": float(prediction.get("score", 0.0)),
            })

    return flattened


def olala_bad_hosts_filter(
    predictions: List[Dict[str, Any]],
    source_ontology: List[Dict[str, Any]],
    target_ontology: List[Dict[str, Any]],
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filters correspondences by expected source and target hosts.

    Parameters:
        predictions (List[Dict[str, Any]]): The correspondences.
        source_ontology (List[Dict[str, Any]]): The encoded source ontology.
        target_ontology (List[Dict[str, Any]]): The encoded target ontology.
        strict (bool): Whether unknown hosts should be removed.

    Returns:
        List[Dict[str, Any]]: The host-filtered correspondences.
    """
    source_host = ""
    target_host = ""

    if source_ontology:
        source_host = source_ontology[0].get("expected_host", "")
    if target_ontology:
        target_host = target_ontology[0].get("expected_host", "")

    return [
        prediction
        for prediction in predictions
        if keep_correspondence_by_host(
            source_iri=prediction["source"],
            target_iri=prediction["target"],
            expected_source_host=source_host,
            expected_target_host=target_host,
            strict=strict,
        )
    ]


def add_alignment_matcher(
    input_predictions: List[Dict[str, Any]],
    alignment_to_add: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Adds one alignment to another with priority for the added alignment.

    Parameters:
        input_predictions (List[Dict[str, Any]]): The input correspondences.
        alignment_to_add (List[Dict[str, Any]]): The correspondences to add.

    Returns:
        List[Dict[str, Any]]: The merged correspondences.
    """
    merged = {}

    for prediction in alignment_to_add:
        pair = (prediction["source"], prediction["target"])
        merged[pair] = prediction

    for prediction in input_predictions:
        pair = (prediction["source"], prediction["target"])
        if pair not in merged:
            merged[pair] = prediction

    return list(merged.values())


def naive_descending_extractor(
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extracts a one-to-one alignment using descending confidence order.

    Parameters:
        predictions (List[Dict[str, Any]]): The correspondences.

    Returns:
        List[Dict[str, Any]]: The one-to-one correspondences.
    """
    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: (
            -float(prediction.get("score", 0.0)),
            prediction["source"],
            prediction["target"],
        ),
    )

    source_matches = set()
    target_matches = set()
    selected = []

    for prediction in sorted_predictions:
        source = prediction["source"]
        target = prediction["target"]

        if source in source_matches or target in target_matches:
            continue

        source_matches.add(source)
        target_matches.add(target)
        selected.append(prediction)

    return selected


def max_weight_bipartite_extractor(
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extracts a maximum-weight one-to-one alignment.

    Parameters:
        predictions (List[Dict[str, Any]]): The correspondences.

    Returns:
        List[Dict[str, Any]]: The one-to-one correspondences.
    """
    if not predictions:
        return predictions

    distinct_confidences = {
        round(float(prediction.get("score", 0.0)), 12)
        for prediction in predictions
    }

    if len(distinct_confidences) == 1:
        return naive_descending_extractor(predictions)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return naive_descending_extractor(predictions)

    source_iris = sorted({prediction["source"] for prediction in predictions})
    target_iris = sorted({prediction["target"] for prediction in predictions})

    source_to_index = {
        source: index
        for index, source in enumerate(source_iris)
    }
    target_to_index = {
        target: index
        for index, target in enumerate(target_iris)
    }

    weight_matrix = np.zeros((len(source_iris), len(target_iris)))
    pair_to_prediction = {}

    for prediction in predictions:
        source = prediction["source"]
        target = prediction["target"]
        score = float(prediction.get("score", 0.0))
        pair = (source, target)

        existing = pair_to_prediction.get(pair)
        if existing is None or score > float(existing.get("score", 0.0)):
            pair_to_prediction[pair] = {
                "source": source,
                "target": target,
                "score": score,
            }
            weight_matrix[
                source_to_index[source],
                target_to_index[target],
            ] = score

    row_indexes, column_indexes = linear_sum_assignment(-weight_matrix)

    selected = []
    for row_index, column_index in zip(row_indexes, column_indexes):
        source = source_iris[row_index]
        target = target_iris[column_index]
        pair = (source, target)

        if pair in pair_to_prediction:
            selected.append(pair_to_prediction[pair])

    return selected


def confidence_filter(
    predictions: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Filters correspondences by confidence threshold.

    Parameters:
        predictions (List[Dict[str, Any]]): The correspondences.
        threshold (float): The confidence threshold.

    Returns:
        List[Dict[str, Any]]: The filtered correspondences.
    """
    return [
        prediction
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= threshold
    ]


def olala_postprocessor(
    alignments: List[Dict[str, Any]],
    encoded_ontology: List[List[Dict[str, Any]]],
    confidence_threshold: float = 0.5,
    strict_bad_hosts: bool = False,
) -> List[Dict[str, Any]]:
    """
    Applies the final OLaLa postprocessing pipeline.

    Parameters:
        llm_predictions (List[Dict[str, Any]]): The LLM-verified predictions.
        hp_predictions (List[Dict[str, Any]]): The high-precision predictions.
        source_ontology (List[Dict[str, Any]]): The encoded source ontology.
        target_ontology (List[Dict[str, Any]]): The encoded target ontology.
        confidence_threshold (float): The final confidence threshold.
        strict_bad_hosts (bool): Whether unknown hosts should be removed.

    Returns:
        List[Dict[str, Any]]: The final postprocessed correspondences.
    """
    source_ontology, target_ontology = encoded_ontology[0], encoded_ontology[1]
    llm_alignments = [prediction for prediction in alignments if prediction.get("atype") == "rag"]
    hp_alignments = [prediction for prediction in alignments if prediction.get("atype") == "hp"]
    llm_alignments = flatten_predictions(llm_alignments)
    hp_alignments = flatten_predictions(hp_alignments)
    hp_alignments = olala_bad_hosts_filter(
        predictions=hp_alignments,
        source_ontology=source_ontology,
        target_ontology=target_ontology,
        strict=strict_bad_hosts,
    )
    merged_alignments = add_alignment_matcher(
        input_predictions=llm_alignments,
        alignment_to_add=hp_alignments,
    )
    one_to_one_predictions = max_weight_bipartite_extractor(merged_alignments)
    return confidence_filter(
        predictions=one_to_one_predictions,
        threshold=confidence_threshold,
    )

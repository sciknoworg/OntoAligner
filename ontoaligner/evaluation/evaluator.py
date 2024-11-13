# -*- coding: utf-8 -*-
"""
This script defines functions for evaluating machine learning models based on different tracks and approaches.

It includes two main functions:
1. `evaluator`: This function performs evaluations by comparing predictions to reference data, handling different track types (e.g., 'bio-ml', 'bio-llm').
2. `evaluator_module`: A more flexible evaluation function that adjusts the evaluation based on the chosen approach (e.g., retrieval, RAG, ICV, fewshot) and processes the predictions accordingly before passing them to the evaluator function.
"""
from typing import Any, Dict, List

from .metrics import evaluation_report
from ..postprocess import process

def evaluator(track: str, predicts: List, references: Any):
    """
    Evaluates predictions against reference data based on the specified track.

    This function handles different tracks like bio-ML and bio-LLM by selecting the relevant subset
    of reference data and applying the evaluation report function.

    Parameters:
        track (str): The track type, determining how the evaluation is carried out (e.g., 'bio-ml', 'bio-llm').
        predicts (List): A list of predicted outputs to be evaluated.
        references (Any): The reference data, which could vary depending on the track, typically in dictionary format.

    Returns:
        results (Dict): A dictionary containing the evaluation results, typically including scores such as precision, recall, and F1.
    """
    if track.startswith("bio-ml"):
        results = {
            "full": evaluation_report(predicts=predicts, references=references["equiv"]["full"]),
            "test": evaluation_report(predicts=predicts, references=references["equiv"]["test"]),
            "train": evaluation_report(predicts=predicts, references=references["equiv"]["train"]),
        }
    elif track.startswith("bio-llm"):
        new_reference = [ref for ref in references["test-cands"] if ref["target"] != "UnMatched"]
        results = evaluation_report(predicts=predicts, references=new_reference)
    else:
        results = evaluation_report(predicts=predicts, references=references)
    return results


def evaluator_module(track: str, approach: str, predicts: List, references: Any, llm_confidence_th: float = 0.7) -> Dict:
    """
    A flexible evaluation function that supports different approaches, including retrieval, RAG, ICV, and few-shot learning.
    It processes predictions before passing them to the evaluator function for evaluation.

    Parameters:
        track (str): The track type, determining how the evaluation is carried out (e.g., 'bio-ml', 'bio-llm').
        approach (str): The evaluation approach to be used, such as 'retrieval', 'rag', 'icv', or 'fewshot'.
        predicts (List): A list of predicted outputs to be evaluated.
        references (Any): The reference data for evaluation.
        llm_confidence_th (float, optional): A threshold for the confidence level in LLM-based approaches. Default is 0.7.

    Returns:
        results (Dict): A dictionary containing the evaluation results and, for certain approaches, additional configuration data.
    """
    if approach == "retrieval":
        predicts = process.eval_preprocess_ir_outputs(predicts=predicts)
    elif approach in ["rag" , "icv", "fewshot"]:
        predicts, configs = process.postprocess_hybrid(predicts=predicts, llm_confidence_th=llm_confidence_th)
    results = evaluator(track=track, predicts=predicts, references=references)
    if approach in ["rag" , "icv", "fewshot"]:
        results = {**results, **configs}
    return results

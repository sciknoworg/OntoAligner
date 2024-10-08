# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from .metrics import evaluation_report
from ..postprocess import process


def evaluator(track: str, predicts: List, references: Any):
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
    if approach == "retrieval":
        predicts = process.eval_preprocess_ir_outputs(predicts=predicts)
    elif approach in ["rag" , "icv", "fewshot"]:
        predicts, configs = process.postprocess_hybrid(predicts=predicts, llm_confidence_th=llm_confidence_th)
    results = evaluator(track=track, predicts=predicts, references=references)
    if approach in ["rag" , "icv", "fewshot"]:
        results = {**results, **configs}
    return results

# -*- coding: utf-8 -*-
from typing import Dict, List


def calculate_intersection(predicts: List, references: List) -> int:
    """
    :param predicts:
        [{
            "source": ...,
            "target": ...,
            "score": ...
        }, ...]
    :param references:
        [{
            "source": ...,
            "target": ...,
            "relation": ...
        }, ...]
    :return: intersection
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
        Calculate Precision Score:
            P = |intersection between out, and ref|/|out|
        Calculate Recall score:
            R = |intersection between out, and ref|/|ref|
        Calculate F-Score:
            F_beta = (1+Beta^2) P*R / (beta^2*P + R)
    :param predicts:
        [{
            "source": ...,
            "target": ...,
            "score": ...
        }, ...]
    :param references:
        [{
            "source": ...,
            "target": ...,
            "relation": ...
        }, ...]
    :return:
        {"intersections":}
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

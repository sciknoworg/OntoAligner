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
This script contains functions to preprocess, evaluate, and filter outputs generated
by information retrieval (IR) systems and language models (LLMs), including confidence
scoring, thresholding, and output filtering. It is used to refine the quality of predictions
by integrating information from both IR and LLM systems, ensuring the most relevant and
confident predictions are retained.

Functions:
    retriever_postprocessor: Prepares IR outputs for further processing by removing irrelevant data.
    llm_postprocessor: Prepares LLM outputs for further processing by removing irrelevant data.
    rag_heuristic_postprocessor: Processes and filters predictions using heuristic methods for confidence scoring.
    rag_hybrid_postprocessor: Hybrid method for processing predictions by integrating IR and LLM results using matrix-based analysis.
"""

from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

from .util import build_outputdict, threshold_finder, confidence_score_ratio_based_filtering, confidence_score_based_filtering


def retriever_postprocessor(predicts: List, threshold: float = 0.0) -> List:
    """
    Prepares IR outputs by extracting source-target pairs and filtering based on score values.

    Parameters:
        predicts (List): List of dictionaries containing source, target candidates, and score candidates.

    Returns:
        List: A list of dictionaries containing source-target pairs with positive scores.
    """
    predicts_temp = []
    for predict in tqdm(predicts):
        source, target_cands, score_cands = predict["source"], predict["target-cands"], predict["score-cands"]
        for target, score in zip(target_cands, score_cands):
            if score > threshold:
                predicts_temp.append({"source": source, "target": target, "score": score})
    return predicts_temp

def llm_postprocessor(predicts: List, mapper:Any, dataset: Any, interested_class: str = 'yes') -> List:
    mapper.fit()
    predicts = mapper.predict(predicts)
    predicts_temp = []
    for predict, data in zip(predicts, dataset):
        if predict == interested_class:
            predicts_temp.append({"source": data["iris"][0], "target": data["iris"][1]})
    return predicts_temp


def rag_heuristic_postprocessor(predicts: List, topk_confidence_ratio: int = 3, topk_confidence_score: int = 1) -> [List, Dict]:
    """
    Processes the predictions using heuristic methods for filtering based on confidence ratio,
    IR score, and LLM confidence score.

    Parameters:
        predicts (List): List of prediction outputs containing both IR and LLM results.
        topk_confidence_ratio (int, optional): Number of top predictions to retain based on confidence ratio.
        topk_confidence_score (int, optional): Number of top predictions to retain based on LLM confidence score.

    Returns:
        List: Filtered predictions after applying the heuristic method.
        Dict: Configuration settings used for filtering predictions.
    """
    ir_outputs = predicts[0]["ir-outputs"]
    llm_outputs = predicts[1]["llm-output"]

    ir_outputs = retriever_postprocessor(predicts=ir_outputs)
    outputdict = build_outputdict(llm_outputs=llm_outputs, ir_outputs=ir_outputs)

    cr_threshold = threshold_finder(outputdict, index=3, use_lst=False)  # 3=confidence_ratio index
    outputdict_confidence_ratios = confidence_score_ratio_based_filtering(
        outputdict=outputdict,
        topk_confidence_ratio=topk_confidence_ratio,
        cr_threshold=cr_threshold,
    )

    ir_score_threshold = threshold_finder(outputdict_confidence_ratios, index=1, use_lst=True)  # 1=ir_score index
    llm_confidence_threshold = threshold_finder(outputdict, index=2, use_lst=False)  # 2=confidence_ratio index

    filtered_predicts = confidence_score_based_filtering(
        outputdict_confidence_ratios=outputdict_confidence_ratios,
        topk_confidence_score=topk_confidence_score,
        llm_confidence_threshold=llm_confidence_threshold,
        ir_score_threshold=ir_score_threshold,
    )
    configs = {
        "topk-confidence-ratio": topk_confidence_ratio,
        "topk-confidence-score": topk_confidence_score,
        "confidence-ratio-th": cr_threshold,
        "ir-score-th": ir_score_threshold,
        "llm-confidence-th": llm_confidence_threshold,
    }
    return filtered_predicts, configs


def rag_hybrid_postprocessor(predicts: List, ir_score_threshold: float = 0.9, llm_confidence_th: float = 0.7) -> [List, Dict]:
    """
    A hybrid approach that integrates IR and LLM outputs using matrix analysis and confidence thresholds.

    Parameters:
        predicts (List): List containing IR and LLM output predictions.
        ir_score_threshold (float, optional): Threshold for IR score filtering. Default is 0.9.
        llm_confidence_th (float, optional): Threshold for LLM confidence score filtering. Default is 0.7.

    Returns:
        List: A list of filtered predictions.
        Dict: A dictionary of configuration parameters used for filtering.
    """
    ir_outputs = predicts[0]["ir-outputs"]
    llm_outputs = predicts[1]["llm-output"]
    ir_cleaned_outputs_id = []
    ir_cleaned_outputs = []
    for ir in ir_outputs:
        if ir["source"] not in ir_cleaned_outputs_id:
            ir_cleaned_outputs_id.append(ir["source"])
            ir_cleaned_outputs.append(ir)
    ir_outputs = ir_cleaned_outputs
    # ir_outputs = retriever_postprocessor(predicts=ir_outputs)
    targets = [target for index, ir_output in enumerate(ir_outputs) for target in ir_output["target-cands"]]
    targets = list(set(targets))
    target2index = {target: index for index, target in enumerate(targets)}
    source2index = {ir_output["source"]: index for index, ir_output in enumerate(ir_outputs)}

    ir_dict = {ir_output["source"]: ir_output for ir_output in ir_outputs}
    ir_dict_based_llm = {ir_output["source"]: np.zeros(len(target2index)) for ir_output in ir_outputs}
    llm_dict = {ir_output["source"]: np.zeros(len(target2index)) for ir_output in ir_outputs}

    outputdict = {}
    for llm_output in tqdm(llm_outputs):
        if llm_output["source"] not in list(outputdict.keys()):
            outputdict[llm_output["source"]] = []
        if llm_output["score"] > llm_confidence_th:
            ir_output = ir_dict.get(llm_output["source"])
            for index, (ir_cand, ir_cand_score) in enumerate(zip(ir_output["target-cands"], ir_output["score-cands"])):
                if ir_cand == llm_output["target"]:
                    confidence_ratio = (llm_output["score"] * 0.2 + 0.8 * ir_cand_score) / 2
                    predicts_list = [llm_output["target"], ir_cand_score, llm_output["score"], confidence_ratio]
                    outputdict[llm_output["source"]].append(predicts_list)
                    ir_dict_based_llm[llm_output["source"]][target2index[ir_cand]] = ir_cand_score
                    llm_dict[llm_output["source"]][target2index[ir_cand]] = confidence_ratio
                    break

    ir_matrix = np.zeros((len(source2index), len(target2index)))

    for iri, output in ir_dict.items():
        for ir_cand, ir_score in zip(output["target-cands"], output["score-cands"]):
            ir_matrix[source2index[iri], target2index[ir_cand]] = ir_score

    ir_matrix_based_llm = np.zeros((len(source2index), len(target2index)))
    for iri, output in ir_dict_based_llm.items():
        ir_matrix_based_llm[source2index[iri], :] = output

    llm_matrix = np.zeros((len(source2index), len(target2index)))
    for iri, output in llm_dict.items():
        llm_matrix[source2index[iri], :] = output

    for col_idx in range(ir_matrix_based_llm.shape[1]):
        col = ir_matrix_based_llm[:, col_idx]
        max_index = np.argmax(col)
        ir_matrix_based_llm[:, col_idx] = np.where(np.arange(len(col)) != max_index, 0, col)

    for col_idx in range(llm_matrix.shape[1]):
        col = llm_matrix[:, col_idx]
        max_index = np.argmax(col)
        llm_matrix[:, col_idx] = np.where(np.arange(len(col)) != max_index, 0, col)

    for row_idx in range(ir_matrix_based_llm.shape[0]):
        row = ir_matrix_based_llm[row_idx, :]
        max_index = np.argmax(row)
        ir_matrix_based_llm[row_idx, :] = np.where(np.arange(len(row)) != max_index, 0, row)

    for row_idx in range(llm_matrix.shape[0]):
        row = llm_matrix[row_idx, :]
        max_index = np.argmax(row)
        llm_matrix[row_idx, :] = np.where(np.arange(len(row)) != max_index, 0, row)

    index2source = {index: source for source, index in source2index.items()}
    index2target = {index: target for target, index in target2index.items()}
    rows, cols = ir_matrix_based_llm.nonzero()
    final_predict = []
    for row, col in zip(rows, cols):
        if ir_matrix_based_llm[row, col] >= ir_score_threshold:
            final_predict.append({
                "source": index2source[row],
                "target": index2target[col],
                "score": ir_matrix_based_llm[row, col]
            })
    configs = {
        "llm-confidence-th": llm_confidence_th,
        "ir-score-th": ir_score_threshold,
    }
    return final_predict, configs

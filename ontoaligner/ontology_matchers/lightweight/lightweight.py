# -*- coding: utf-8 -*-
from typing import Any, List

import rapidfuzz
from tqdm import tqdm

from ontoaligner.base import BaseOMModel


class Lightweight(BaseOMModel):
    path: str = "NO MODEL LOADING IN LIGHTWEIGHT MODEL"

    def __str__(self):
        return "Lightweight"

    def init_retriever(self, data):
        pass

    def generate(self, input_data: List) -> List:
        # source_onto, target_onto
        # index into retriever
        pass


class FuzzySMLightweight(Lightweight):
    """
    Fuzzy String Matching using: https://github.com/maxbachmann/RapidFuzz#partial-ratio
    """

    def ratio_estimate(self) -> Any:
        pass

    def calculate_similarity(self, source: str, candidates: List) -> [int, float]:
        selected_candid = rapidfuzz.process_cpp.extractOne(
            source,
            candidates,
            scorer=self.ratio_estimate(),
            processor=rapidfuzz.utils.default_process,
        )
        return selected_candid[2], selected_candid[1] / 100

    def generate(self, input_data: List) -> List:
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []
        candidates = [target["text"] for target in target_ontology]
        for source in tqdm(source_ontology):
            selected_candid_idx, selected_candid_score = self.calculate_similarity(source=source["text"],
                                                                                   candidates=candidates)
            if selected_candid_score >= self.kwargs["fuzzy_sm_threshold"]:
                predictions.append(
                    {
                        "source": source["iri"],
                        "target": target_ontology[selected_candid_idx]["iri"],
                        "score": selected_candid_score,
                    }
                )
        return predictions

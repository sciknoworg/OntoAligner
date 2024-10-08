# -*- coding: utf-8 -*-
import time
from typing import Any


class OMPipelines:
    def __init__(self,
                 owl_json_path: str=None,
                 llm_confidence_th: float=0.7,
                 ir_score_threshold: float=0.9,
                 output_dir: str=None,
                 ontology_matcher: Any=None,
                 om_dataset: Any=None,
                 om_encoder: Any=None,
                 **kwargs) -> None:
        self.owl_json_path = owl_json_path
        self.llm_confidence_th = llm_confidence_th
        self.ir_score_threshold = ir_score_threshold
        self.output_dir = output_dir
        self.kwargs = kwargs
        self.om_dataset = om_dataset
        self.ontology_matcher = ontology_matcher
        self.om_encoder = om_encoder()

    def __call__(self) -> None:
        task_obj = self.om_dataset()
        if self.owl_json_path:
            task_owl = task_obj.load_from_json(root_dir=self.owl_json_path)
        else:
            task_owl = task_obj.collect(root_dir=self.config.root_dir)
        output_dict_obj = {
            "dataset-info": task_owl["dataset-info"],
            "encoder-info": self.om_encoder.get_encoder_info(),
        }
        encoded_inputs = self.om_encoder(**task_owl)
        print("\t\tWorking on generating response!")
        start_time = time.time()
        model_output = self.ontology_matcher.generate(input_data=encoded_inputs)
        output_dict_obj["response-time"] = time.time() - start_time
        output_dict_obj["generated-output"] = model_output

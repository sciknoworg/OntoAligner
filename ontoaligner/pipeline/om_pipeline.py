# -*- coding: utf-8 -*-
"""
This script defines the OMPipelines class, which handles the process of ontology matching by
loading source and target ontologies, processing them through a dataset object, and generating
matching results using an ontology matcher. The class also manages the integration of an encoder
and handles parameters related to ontology matching, including thresholds for confidence and similarity.

Functionality:
    - Initializes with paths to ontologies and other optional configurations such as confidence
      thresholds, output directories, and external modules (ontology matcher, dataset, encoder).
    - Executes ontology matching by either loading from a JSON file or collecting data from
      source and target ontologies.
    - Encodes inputs using the specified encoder and generates matching results using the ontology matcher.
    - Returns a dictionary containing dataset info, encoder info, response time, and the model output.

Classes:
    - OMPipelines: A class for orchestrating the ontology matching pipeline using external dataset
      loading, encoding, and matching components.
"""

import time
from typing import Any


class OMPipelines:
    """
    This class defines a pipeline for ontology matching, where it takes in source and target ontologies,
    processes them through a dataset object, and uses an ontology matcher to generate matching results.

    Attributes:
        source_ontology_path (str): Path to the source ontology.
        target_ontology_path (str): Path to the target ontology.
        reference_matching_path (str, optional): Path to reference matching data. Default is None.
        owl_json_path (str, optional): Path to the owl JSON file. Default is None.
        llm_confidence_th (float): Confidence threshold for the LLM. Default is 0.7.
        ir_score_threshold (float): Information retrieval score threshold. Default is 0.9.
        output_dir (str, optional): Directory where the output will be stored. Default is None.
        kwargs (dict): Additional optional keyword arguments.
        om_dataset (Any): Dataset object used to handle the ontology data.
        ontology_matcher (Any): Matcher used to perform ontology matching.
        om_encoder (Any): Encoder used to encode the ontology data.

    Methods:
        __call__(self) -> None:
            Executes the ontology matching pipeline by loading the data, encoding it, and generating results.
    """

    def __init__(self,
                 source_ontology_path: str,
                 target_ontology_path: str,
                 reference_matching_path: str = None,
                 owl_json_path: str = None,
                 llm_confidence_th: float = 0.7,
                 ir_score_threshold: float = 0.9,
                 output_dir: str = None,
                 ontology_matcher: Any = None,
                 om_dataset: Any = None,
                 om_encoder: Any = None,
                 **kwargs) -> None:
        """
        Initializes the OMPipelines class with the required and optional parameters.

        Parameters:
            source_ontology_path (str): Path to the source ontology.
            target_ontology_path (str): Path to the target ontology.
            reference_matching_path (str, optional): Path to reference matching data (default is None).
            owl_json_path (str, optional): Path to the owl JSON file (default is None).
            llm_confidence_th (float): Confidence threshold for the LLM (default is 0.7).
            ir_score_threshold (float): Information retrieval score threshold (default is 0.9).
            output_dir (str, optional): Directory where the output will be stored (default is None).
            ontology_matcher (Any, optional): Matcher used for ontology matching (default is None).
            om_dataset (Any, optional): Dataset object for handling ontology data (default is None).
            om_encoder (Any, optional): Encoder for encoding ontology data (default is None).
            **kwargs (dict): Additional keyword arguments.
        """
        self.source_ontology_path = source_ontology_path
        self.target_ontology_path = target_ontology_path
        self.reference_matching_path = reference_matching_path
        self.owl_json_path = owl_json_path
        self.llm_confidence_th = llm_confidence_th
        self.ir_score_threshold = ir_score_threshold
        self.output_dir = output_dir
        self.kwargs = kwargs
        self.om_dataset = om_dataset
        self.ontology_matcher = ontology_matcher
        self.om_encoder = om_encoder()

    def __call__(self) -> None:
        """
        Executes the ontology matching process. This includes loading data from files,
        processing the ontologies through the dataset object, encoding the inputs using
        the encoder, and generating matching results with the ontology matcher.

        Process:
            1. Loads ontology data from a JSON file or collects from the source/target paths.
            2. Encodes the input data using the specified encoder.
            3. Generates ontology matching results.
            4. Outputs the results along with relevant information such as the response time.

        Returns:
            None
        """
        task_obj = self.om_dataset()
        if self.owl_json_path:
            task_owl = task_obj.load_from_json(root_dir=self.owl_json_path)
        else:
            task_owl = task_obj.collect(source_ontology_path=self.source_ontology_path,
                                        target_ontology_path=self.target_ontology_path,
                                        reference_matching_path=self.reference_matching_path)
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

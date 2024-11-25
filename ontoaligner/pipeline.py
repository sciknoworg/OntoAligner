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
from .base import BaseOMModel
from .utils import metrics, xmlify

class Pipeline:
    """
    This class defines a pipeline for ontology matching, where it takes in source and target ontologies,
    processes them through a dataset object, and uses an ontology matcher to generate matching results.

    Attributes:
        ontology_matcher (Any): Matcher used to perform ontology matching.
        om_encoder (Any): Encoder used to encode the ontology data.
        kwargs (dict): Additional optional keyword arguments.
    """

    def __init__(self,
                 ontology_matcher: BaseOMModel,
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
        self.kwargs = kwargs
        self.ontology_matcher = ontology_matcher
        self.om_encoder = om_encoder()

    def __call__(self,
                 source_ontology_path: str,
                 target_ontology_path: str,
                 owl_json_path: str = None,
                 om_dataset: Any = None,
                 reference_matching_path: str = None,
                 llm_confidence_th: float = 0.7,
                 ir_score_threshold: float = 0.9,
                 evaluation: bool = False,
                 return_dict: bool = False,
                 return_rdf: bool=False,
                 relation: str = "=",
                 digits: int=-2) -> Any:
        """
        Executes the ontology matching process. This includes loading data from files,
        processing the ontologies through the dataset object, encoding the inputs using
        the encoder, and generating matching results with the ontology matcher.

        Process:
            1. Loads ontology data from a JSON file or collects from the source/target paths.
            2. Encodes the input data using the specified encoder.
            3. Generates ontology matching results.
            4. Outputs the results along with relevant information such as the response time.
        Parameters:
            source_ontology_path (str): Path to the source ontology.
            target_ontology_path (str): Path to the target ontology.
            reference_matching_path (str, optional): Path to reference matching data (default is None).
            owl_json_path (str, optional): Path to the owl JSON file (default is None).
            llm_confidence_th (float): Confidence threshold for the LLM (default is 0.7).
            ir_score_threshold (float): Information retrieval score threshold (default is 0.9).
            om_dataset (Any, optional): Dataset object for handling ontology data (default is None).
        Returns:
            Matched ontologies
        """
        task = om_dataset()
        if owl_json_path:
            dataset = task.load_from_json(root_dir=owl_json_path)
        else:
            dataset = task.collect(source_ontology_path=source_ontology_path,
                                   target_ontology_path=target_ontology_path,
                                   reference_matching_path=reference_matching_path)
        output_dict = {
            "dataset-info": dataset["dataset-info"],
            "encoder-info": self.om_encoder.get_encoder_info(),
        }
        encoder_output = self.om_encoder(source=dataset['source'], target=dataset['target'])
        print("\t\tWorking on generating response!")
        start_time = time.time()
        model_output = self.ontology_matcher.generate(input_data=encoder_output)
        output_dict["response-time"] = time.time() - start_time
        output_dict["generated-output"] = model_output
        if evaluation:
            output_dict['evaluation'] =metrics.evaluation_report(predicts=model_output,
                                                                 references=dataset['reference'])
        if return_dict:
            return output_dict
        else:
            xlm = xmlify.xml_alignment_generator(matchings=model_output,
                                                 return_rdf=return_rdf,
                                                 relation=relation,
                                                 digits=digits)
            return xlm

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
This module consists of two pipeline utilities: OntoAlignerPipeline and AlignerPipeline.

Ontology Alignment Pipeline. Various methods such as lightweight matching, retriever-based matching, LLM-based matching,
and RAG (Retriever-Augmented Generation) techniques has been applied.

AlignerPipeline runs user-provided encoder, aligner and optional postprocessor components over a collected ontology matching dataset. Unlike
OntoAlignerPipeline, it does not collect datasets, select methods, evaluate results, or save outputs.
"""
import json
import inspect
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Any, Dict, List
from sklearn.linear_model import LogisticRegression

from .base import BaseEncoder, BaseOMModel, OMDataset
from .encoder import (ConceptLightweightEncoder,
                      ConceptLLMEncoder,
                      ConceptRAGEncoder,
                      ConceptParentFewShotEncoder,
                      GraphTripleEncoder)
from .utils import metrics, xmlify
from .aligner import (SimpleFuzzySMLightweight,
                      SBERTRetrieval,
                      AutoModelDecoderLLM,
                      ConceptLLMDataset)
from .postprocess import (retriever_postprocessor,
                          llm_postprocessor,
                          rag_hybrid_postprocessor,
                          TFIDFLabelMapper,
                          LabelMapper)


class AlignerPipeline(BaseOMModel):
    """
    An aligner pipeline that runs one encoder and one ontology matching aligner.

    This class follows the standard OntoAligner flow for one aligner pipeline:
    encode the ontology matching dataset, load the aligner if needed, generate predictions,
    and optionally apply a postprocessor.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        aligner: BaseOMModel,
        om_dataset: Dict = None,
        load_params: Dict = None,
        llm_dataset_class: Dataset = None,
        batch_size: int = 1,
        shuffle: bool = False,
        postprocessor: Any = None,
        postprocessor_params: Dict = None,
        include_reference: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the aligner pipeline.

        Parameters:
            encoder (BaseEncoder): Encoder model used to encode the ontology matching dataset.
            aligner (BaseOMModel): Ontology matching aligner used to generate predictions.
            om_dataset (Dict, optional): Pre-collected ontology matching dataset. Defaults to None.
            load_params (Dict, optional): Parameters forwarded to the aligner load method. Defaults to None.
            llm_dataset_class (Dataset, optional): Dataset class used to wrap LLM inputs. Defaults to None.
            batch_size (int, optional): Batch size used for LLM dataset generation. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle LLM dataset batches. Defaults to False.
            postprocessor (Any, optional): Optional postprocessor applied to predictions. Defaults to None.
            postprocessor_params (Dict, optional): Optional parameters forwarded to the postprocessor. Defaults to None.
            include_reference (bool, optional): Whether to pass reference matchings to the encoder. Defaults to False.
            **kwargs: Additional keyword arguments that may be used for model configuration.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.aligner = aligner
        self.om_dataset = om_dataset
        self.load_params = load_params or {}
        self.llm_dataset_class = llm_dataset_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.postprocessor = postprocessor
        self.postprocessor_params = postprocessor_params or {}
        self.include_reference = include_reference

    def __str__(self):
        """
        Returns a string representation of the AlignerPipeline model.

        Returns:
            str: A simple string representation of the class ("AlignerPipeline").
        """
        return "AlignerPipeline"

    def _is_graph_encoder(self) -> bool:
        """
        Checks whether the pipeline's encoder is a graph triple encoder.

        Returns:
            bool: True if the encoder is a GraphTripleEncoder, otherwise False.
        """
        return isinstance(self.encoder, GraphTripleEncoder)

    def _needs_llm_dataset(self) -> bool:
        """
        Checks whether the encoded output needs to be wrapped as an LLM dataset.

        Returns:
            bool: True if an LLM dataset class is provided, otherwise False.
        """
        return self.llm_dataset_class is not None

    def _encode(self, om_dataset: Dict) -> List:
        """
        Encodes the ontology matching dataset.

        Parameters:
            om_dataset (Dict): The ontology matching dataset.

        Returns:
            List: The encoded ontology matching data.
        """
        if self._is_graph_encoder():
            return self.encoder(**om_dataset)

        if self.include_reference:
            return self.encoder(
                source=om_dataset["source"],
                target=om_dataset["target"],
                reference=om_dataset["reference"],
            )

        return self.encoder(
            source=om_dataset["source"],
            target=om_dataset["target"],
        )

    def _load_aligner(self) -> None:
        """
        Loads the ontology matching aligner when load parameters are provided.
        """
        if hasattr(self.aligner, "load") and self.load_params:
            self.aligner.load(**self.load_params)

    def _generate_llm_predictions(self, llm_dataset: Dataset) -> List:
        """
        Generates LLM predictions from an LLM dataset using batched prompts.

        Parameters:
            llm_dataset (Dataset): Dataset containing LLM prompts and source-target IRIs.

        Returns:
            List: A list of generated LLM outputs.
        """
        dataloader = DataLoader(
            llm_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=llm_dataset.collate_fn,
        )

        predictions = []
        for batch in tqdm(dataloader):
            sequences = self.aligner.generate(batch["prompts"])
            predictions.extend(sequences)

        return predictions

    def _apply_postprocessor(self, predictions: List, llm_dataset: Dataset = None) -> List:
        """
        Applies the optional postprocessor to pipeline predictions.

        Parameters:
            predictions (List): The generated pipeline predictions.
            llm_dataset (Dataset, optional): LLM dataset used to generate prompts. Defaults to None.

        Returns:
            List: The postprocessed predictions.
        """
        postprocessor_params = dict(self.postprocessor_params)

        if llm_dataset is not None and "dataset" not in postprocessor_params:
            postprocessor_params["dataset"] = llm_dataset

        signature = inspect.signature(self.postprocessor)
        parameters = signature.parameters

        if "predicts" in parameters:
            predictions = self.postprocessor(
                predicts=predictions,
                **postprocessor_params,
            )
        else:
            predictions = self.postprocessor(
                predictions,
                **postprocessor_params,
            )

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        return predictions

    def generate(self, input_data: Dict = None) -> List:
        """
        Generates predictions for one aligner pipeline.

        Parameters:
            input_data (Dict, optional): Optional ontology matching dataset. If not provided,
                                         the pipeline uses its own pre-collected dataset.

        Returns:
            List: A list of raw or postprocessed alignment predictions.
        """
        llm_dataset = None
        om_dataset = input_data or self.om_dataset

        if om_dataset is None:
            raise ValueError("AlignerPipeline requires an ontology matching dataset.")

        encoded_data = self._encode(om_dataset=om_dataset)
        self._load_aligner()

        if self._needs_llm_dataset():
            llm_dataset = self.llm_dataset_class(
                source_onto=encoded_data[0],
                target_onto=encoded_data[1],
            )
            predictions = self._generate_llm_predictions(llm_dataset=llm_dataset)
        else:
            predictions = self.aligner.generate(input_data=encoded_data)

        if self.postprocessor is not None:
            predictions = self._apply_postprocessor(
                predictions=predictions,
                llm_dataset=llm_dataset,
            )

        return predictions


class OntoAlignerPipeline:
    """
    A pipeline for performing ontology alignment tasks using various methods and models.
    """
    def __init__(self, task_class: OMDataset, source_ontology_path: str, target_ontology_path: str,
                 reference_matching_path: str, output_dir: str ="results", output_format: str ="xml"):
        """
        Initializes the OntoAlignerPipeline.

        Parameters:
            task_class (OMDataset): Class responsible for handling ontology matching tasks.
            source_ontology_path (str): Path to the source ontology file.
            target_ontology_path (str): Path to the target ontology file.
            reference_matching_path (str): Path to the reference alignments.
            output_dir (str, optional): Directory to save results. Defaults to "results".
            output_format (str, optional): Format of output files. Defaults to "xml".
        """
        self.task_class = task_class
        self.source_ontology_path = source_ontology_path
        self.target_ontology_path = target_ontology_path
        self.reference_matching_path = reference_matching_path
        self.output_dir = Path(output_dir)
        self.output_format = output_format.lower()
        self.task = self._initialize_task()
        self.dataset = self._collect_dataset()

    def _initialize_task(self):
        """
        Initializes the ontology matching task.

        Returns:
            OMDataset: Initialized task object.
        """
        return self.task_class()

    def _collect_dataset(self):
        """
        Collects the dataset required for ontology matching.

        Returns:
            dict: A dictionary containing source, target, and reference datasets.
        """
        return self.task.collect(
            source_ontology_path=self.source_ontology_path,
            target_ontology_path=self.target_ontology_path,
            reference_matching_path=self.reference_matching_path
        )

    def __call__(self, method: str, encoder_model: BaseEncoder = None, model_class: BaseOMModel = None, dataset_class: Dataset = None, postprocessor: Any = None,
                 llm_path: str = None, retriever_path: str = None, device: str = "cuda", batch_size: int = 2048, max_length: int = 300, max_new_tokens: int = 10,
                 top_k: int = 10, fuzzy_sm_threshold: float = 0.2, evaluate: bool = False, return_matching: bool = True, output_file_name: str = "matchings",
                 save_matchings: bool = False, ir_threshold: float = 0.5, ir_rag_threshold: float = 0.7, llm_threshold: float = 0.5, llm_mapper: LabelMapper = None,
                 llm_mapper_interested_class: str = 'yes', answer_set: Dict = {"yes": ["yes", "true"], "no": ["no", "false"]}, huggingface_access_token: str = "",
                 openai_key: str = "", device_map: str = "auto", positive_ratio: float = 0.7, n_shots: int = 5) -> [Any, Any]:
        """
        Executes the ontology alignment process using the specified method.

        Parameters:
            method (str): The method to use, e.g., "lightweight", "retriever", or "llm".
            encoder_model (BaseEncoder, optional): Encoder model to encode ontologies. Defaults to None.
            model_class (BaseOMModel, optional): Model class for matching. Defaults to None.
            dataset_class (Dataset, optional): Dataset class for LLM-based methods. Defaults to None.
            postprocessor (Any, optional): Post-processing function. Defaults to None.
            llm_path (str, optional): Path to the LLM model. Defaults to None.
            retriever_path (str, optional): Path to the retriever model. Defaults to None.
            device (str, optional): Device to use for computation. Defaults to "cuda".
            batch_size (int, optional): Batch size for LLM-based methods. Defaults to 2048.
            max_length (int, optional): Maximum input length for LLM-based methods. Defaults to 300.
            max_new_tokens (int, optional): Maximum tokens to generate for LLM-based methods. Defaults to 10.
            top_k (int, optional): Number of top matches to retrieve in the retriever method. Defaults to 10.
            fuzzy_sm_threshold (float, optional): Threshold for fuzzy matching in lightweight methods. Defaults to 0.2.
            evaluate (bool, optional): Whether to evaluate the matching results. Defaults to False.
            return_matching (bool, optional): Whether to return the matching results. Defaults to True.
            output_file_name (str, optional): Output file name without file type. Defaults to "matchings".
            save_matchings (bool, optional): Whether to save the matching results. Defaults to False.
            ir_threshold (float, optional): Retrieval postprocessor threshold.
            ir_rag_threshold (float, optional): Retrieval postprocessor threshold in RAG module.
            llm_threshold (float, optional): LLM postprocessor threshold.
            llm_mapper (LabelMapper, optional): Mapper for LLM outputs.
            llm_mapper_interested_class (str, optional): Class to filter output pairs in LLM postprocessing.
            answer_set (dict, optional): Mapping of yes/no answers. Defaults to {"yes": ["yes", "true"], "no": ["no", "false"]}.
            huggingface_access_token (str, optional): Access token for Hugging Face models. Defaults to "".
            openai_key (str, optional): API key for OpenAI models. Defaults to "".
            device_map (str, optional): Device map for model allocation. Defaults to "auto".
            positive_ratio (float, optional): Ratio of positive examples in few-shot methods. Defaults to 0.7.
            n_shots (int, optional): Number of shots for few-shot learning. Defaults to 5.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        if not (0 <= fuzzy_sm_threshold <= 1):
            raise ValueError(f"fuzzy_sm_threshold must be between 0 and 1. Got {fuzzy_sm_threshold}")

        if method not in ["lightweight", "retrieval", "llm"] and "rag" not in method:
            raise ValueError(f"Unknown method: {method}")

        if method == "lightweight":
            matchings = self._run_lightweight(encoder_model or ConceptLightweightEncoder(), model_class or SimpleFuzzySMLightweight,
                                              postprocessor, fuzzy_sm_threshold)
        if method == "retrieval":
            matchings = self._run_retriever(encoder_model or ConceptLightweightEncoder(), model_class or SBERTRetrieval, postprocessor or retriever_postprocessor,
                                             retriever_path, device, top_k, ir_threshold)
        if method == "llm":
            matchings = self._run_llm(encoder_model or ConceptLLMEncoder(), model_class or AutoModelDecoderLLM, dataset_class or ConceptLLMDataset,
                                       postprocessor or llm_postprocessor, llm_mapper or TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1, 1)),
                                       llm_mapper_interested_class, llm_path, device, batch_size, max_length, max_new_tokens, llm_threshold)
        if 'rag' in method:
            retriever_config = {"device": device, "top_k": top_k, "openai_key": openai_key}
            llm_config = {"device": device, "batch_size": batch_size, "answer_set": answer_set, "huggingface_access_token": huggingface_access_token,
                          "max_length": max_length, "max_new_tokens": max_new_tokens, "openai_key": openai_key, "device_map": device_map}
            rag_config = {"retriever_config": retriever_config, "llm_config": llm_config}
            if method == 'fewshot-rag':
                rag_config['n_shots'] = n_shots
                rag_config['positive_ratio'] = positive_ratio
                encoder_model = encoder_model or ConceptParentFewShotEncoder()
            else:
                encoder_model = encoder_model or ConceptRAGEncoder()
            matchings = self._run_rag(method, encoder_model, model_class, postprocessor or rag_hybrid_postprocessor,
                                      llm_threshold,  ir_rag_threshold, retriever_path, llm_path, rag_config)
        return self._process_results(matchings, method, evaluate, return_matching, output_file_name, save_matchings)

    def _run_lightweight(self, encoder_model, model_class, postprocessor, fuzzy_sm_threshold):
        """
        Executes the lightweight ontology alignment method.

        This method uses a lightweight matching model to generate ontology matchings
        based on encoded source and target ontologies. Optionally, a postprocessor
        is applied to refine the matching results.

        Parameters:
            encoder_model (BaseEncoder): Encoder model to encode the source and target ontologies.
            model_class (BaseOMModel): A class implementing the lightweight matching logic.
            postprocessor (callable or None): A function to refine the matching results. Optional.
            fuzzy_sm_threshold (float): Threshold value for fuzzy similarity matching in the model.

        Returns:
            dict: The resulting matchings after encoding and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        model = model_class(fuzzy_sm_threshold=fuzzy_sm_threshold)
        matchings = model.generate(input_data=encoder_output)
        if postprocessor:
            matchings = postprocessor(matchings)
        return matchings

    def _run_retriever(self, encoder_model, model_class, postprocessor, retriever_path, device, top_k, ir_threshold):
        """
        Executes the retriever-based ontology alignment method.

        This method leverages a retriever model to identify top-k potential matches
        between ontologies based on their encoded representations. The results are
        refined using a postprocessor.

        Parameters:
            encoder_model (BaseEncoder): Encoder model to encode the source and target ontologies.
            model_class (BaseOMModel): A class implementing the retriever-based matching logic.
            postprocessor (callable): A function to refine the matching results.
            retriever_path (str): File path to the pretrained retriever model.
            device (str): The computational device (e.g., 'cpu' or 'cuda').
            top_k (int): Number of top candidate matches to retrieve.
            ir_threshold (float): Threshold for the postprocessor to filter results.

        Returns:
            dict: The resulting matchings after encoding, retrieval, and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        model = model_class(device=device, top_k=top_k)
        model.load(path=retriever_path)
        matchings = model.generate(input_data=encoder_output)
        matchings = postprocessor(matchings, threshold=ir_threshold)
        return matchings

    def _run_llm(self, encoder_model, model_class, dataset_class, postprocessor, llm_mapper,
                 llm_mapper_interested_class,
                 llm_path, device, batch_size, max_length, max_new_tokens, llm_threshold):
        """
        Executes the LLM-based ontology alignment method.

        This method uses a large language model (LLM) to generate matching predictions
        based on a dataset constructed from the encoded ontologies. The results are
        postprocessed to align with the desired format and filtering criteria.

        Parameters:
            encoder_model (BaseEncoder): Encoder model to encode the source and target ontologies.
            model_class (BaseOMModel): A class implementing LLM-based matching logic.
            dataset_class (Dataset): A class to construct datasets for LLM-based methods.
            postprocessor (callable): A function to refine the matching results.
            llm_mapper (LabelMapper): A mapper to process LLM outputs into matchings.
            llm_mapper_interested_class (str): Specific class of matchings to filter in the results.
            llm_path (str): File path to the pretrained LLM model.
            device (str): The computational device (e.g., 'cpu' or 'cuda').
            batch_size (int): Number of samples to process in each batch.
            max_length (int): Maximum input sequence length for the LLM.
            max_new_tokens (int): Maximum tokens to generate for each sequence.
            llm_threshold (float): Threshold for the postprocessor to filter results.

        Returns:
            dict: The resulting matchings after encoding, LLM generation, and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        llm_dataset = dataset_class(source_onto=encoder_output[0], target_onto=encoder_output[1])
        dataloader = DataLoader(llm_dataset, batch_size=batch_size, shuffle=False, collate_fn=llm_dataset.collate_fn)
        model = model_class(device=device, max_length=max_length, max_new_tokens=max_new_tokens)
        model.load(path=llm_path)

        matchings = []
        for batch in tqdm(dataloader):
            sequences = model.generate(batch["prompts"])
            matchings.extend(sequences)

        matchings = postprocessor(predicts=matchings, mapper=llm_mapper, dataset=llm_dataset,
                                  interested_class=llm_mapper_interested_class)
        return matchings

    def _run_rag(self, method, encoder_model, model_class, postprocessor, llm_threshold, ir_threshold, retriever_path,
                 llm_path, rag_config):
        """
        Executes the RAG (Retriever-Augmented Generation) ontology alignment method.

        This method combines retriever-based and LLM-based techniques to generate
        ontology matchings. A retriever identifies candidate matches, and an LLM
        refines the results. The final matchings are postprocessed to meet thresholds.

        Parameters:
            method (str): Specific RAG method to use (e.g., 'icv-rag', 'fewshot-rag').
            encoder_model (BaseEncoder): Encoder model to encode the ontologies.
            model_class (BaseOMModel): A class implementing RAG-based matching logic.
            postprocessor (callable): A function to refine the matching results.
            llm_threshold (float): Confidence threshold for the LLM-based predictions.
            ir_threshold (float): Score threshold for the retriever results.
            retriever_path (str): File path to the retriever model.
            llm_path (str): File path to the LLM model.
            rag_config (dict): Configuration parameters for the RAG model.

        Returns:
            dict: The resulting matchings after encoding, RAG generation, and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'],
                                       target=self.dataset['target'],
                                       reference=self.dataset[
                                           'reference'] if method == 'icv-rag' or method == 'fewshot-rag' else None)
        model = model_class(**rag_config)
        model.load(llm_path=llm_path, ir_path=retriever_path)
        matchings = model.generate(input_data=encoder_output)
        matchings, _ = postprocessor(matchings, ir_score_threshold=ir_threshold, llm_confidence_th=llm_threshold)
        return matchings

    def _process_results(self, matchings, method, evaluate, return_matching, output_file_name, save_matchings):
        """
        Processes and evaluates the matching results.

        Parameters:
            matchings (list): List of matching results.
            method (str): The method used for alignment.
            evaluate (bool): Whether to evaluate the results.
            return_matching (bool): Whether to return the matching results.
            output_file_name (str): Output file name.
            save_matchings (bool, optional):  Whether to save the matching in results or not.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        output_matches = None
        if self.output_format == "xml":
            xml_str = xmlify.xml_alignment_generator(matchings=matchings)
            output_matches = xml_str
        elif self.output_format == "json":
            output_matches = matchings
        else:
            raise ValueError("Unsupported output format")

        if save_matchings:
            output_dir = self.output_dir / method
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.output_format == "xml":
                with open(output_dir / f"{output_file_name}.xml", "w", encoding="utf-8") as xml_file:
                    xml_file.write(output_matches)
            elif self.output_format == "json":
                with open(output_dir / f"{output_file_name}.json", "w", encoding="utf-8") as json_file:
                    json.dump(output_matches, json_file, indent=4, ensure_ascii=False)

        evaluation = None
        if evaluate:
            evaluation = metrics.evaluation_report(predicts=matchings, references=self.dataset['reference'])

        if return_matching:
            if evaluate:
                return matchings, evaluation
            return output_matches
        if evaluate:
            return evaluation

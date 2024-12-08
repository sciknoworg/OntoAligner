import json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Any

# Import necessary modules from the ontoaligner library
from ontoaligner.base import BaseEncoder, BaseOMModel, OMDataset
from ontoaligner.encoder import ConceptLightweightEncoder, ConceptLLMEncoder
from ontoaligner.utils import metrics, xmlify
from ontoaligner.ontology_matchers import SimpleFuzzySMLightweight, SBERTRetrieval, AutoModelDecoderLLM, ConceptLLMDataset
from ontoaligner.postprocess import retriever_postprocessor, llm_postprocessor


class OntoAlignerPipeline:
    """
        A pipeline for performing ontology alignment tasks using various methods and models.
    """
    def __init__(self, task_class: OMDataset, source_ontology_path: str, target_ontology_path: str,
                 reference_matching_path: str, output_path: str ="results", output_format: str ="xml"):
        self.task_class = task_class
        self.source_ontology_path = source_ontology_path
        self.target_ontology_path = target_ontology_path
        self.reference_matching_path = reference_matching_path
        self.output_path = Path(output_path)
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

    def __call__(self, method: str,  encoder_model: BaseEncoder =None, model_class: BaseOMModel=None, dataset_class:Dataset =None, postprocessor: Any=None,
                 llm_path: str=None, retriever_path: str=None, device: str="cuda", batch_size: int=2048, max_length: int=300, max_new_tokens: int=10,
                 top_k: int=10, fuzzy_sm_threshold: float=0.2, evaluate: bool=True, return_matching: bool=False):
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
            evaluate (bool, optional): Whether to evaluate the matching results. Defaults to True.
            return_matching (bool, optional): Whether to return the matching results. Defaults to False.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        if method == "lightweight":
            return self._run_lightweight(encoder_model or ConceptLightweightEncoder(), model_class or SimpleFuzzySMLightweight,
                                         postprocessor, fuzzy_sm_threshold, evaluate, return_matching)
        elif method == "retriever":
            return self._run_retriever(encoder_model or ConceptLightweightEncoder(), model_class or SBERTRetrieval,
                                       postprocessor or retriever_postprocessor, retriever_path, top_k, evaluate, return_matching)
        elif method == "llm":
            return self._run_llm(encoder_model or ConceptLLMEncoder(), model_class or AutoModelDecoderLLM, dataset_class or ConceptLLMDataset,
                                 postprocessor or llm_postprocessor, llm_path, device, batch_size, max_length, max_new_tokens, evaluate, return_matching)
        elif method == 'rag':
            return self._run_rag()
        elif method == 'fewshot':
            return self._run_fewshot_rag()
        elif method == 'icv':
            return self._run_icv_rag()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _run_lightweight(self, encoder_model, model_class, postprocessor, fuzzy_sm_threshold, evaluate, return_matching):
        """
        Runs the lightweight ontology alignment method.

        Parameters:
            encoder_model (BaseEncoder): Encoder model for encoding ontologies.
            model_class (BaseOMModel): Model class for lightweight matching.
            postprocessor (Any): Post-processing function.
            fuzzy_sm_threshold (float): Threshold for fuzzy matching.
            evaluate (bool): Whether to evaluate the results.
            return_matching (bool): Whether to return the matching results.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        model = model_class(fuzzy_sm_threshold=fuzzy_sm_threshold)
        matchings = model.generate(input_data=encoder_output)
        if postprocessor:
            matchings = postprocessor(matchings)
        return self._process_results(matchings, "lightweight", evaluate, return_matching)

    def _run_retriever(self, encoder_model, model_class, postprocessor, retriever_path, top_k, evaluate, return_matching):
        """
        Runs the retriever-based ontology alignment method.

        Parameters:
            encoder_model (BaseEncoder): Encoder model for encoding ontologies.
            model_class (BaseOMModel): Model class for retriever-based matching.
            postprocessor (Any): Post-processing function.
            retriever_path (str): Path to the retriever model.
            top_k (int): Number of top matches to retrieve.
            evaluate (bool): Whether to evaluate the results.
            return_matching (bool): Whether to return the matching results.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        model = model_class(device='cpu', top_k=top_k)
        model.load(path=retriever_path)
        matchings = model.generate(input_data=encoder_output)
        matchings = postprocessor(matchings)
        return self._process_results(matchings, "retriever", evaluate, return_matching)


    def _run_llm(self, encoder_model, model_class, dataset_class, postprocessor, llm_path,
                 device, batch_size, max_length, max_new_tokens, evaluate, return_matching):
        """
        Runs the LLM-based ontology alignment method.

        Parameters:
            encoder_model (BaseEncoder): Encoder model for encoding ontologies.
            model_class (BaseOMModel): Model class for LLM-based matching.
            dataset_class (Dataset): Dataset class for LLM-based methods.
            postprocessor (Any): Post-processing function.
            llm_path (str): Path to the LLM model.
            device (str): Device to use for computation.
            batch_size (int): Batch size for LLM-based methods.
            max_length (int): Maximum input length for LLM-based methods.
            max_new_tokens (int): Maximum tokens to generate.
            evaluate (bool): Whether to evaluate the results.
            return_matching (bool): Whether to return the matching results.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        llm_dataset = dataset_class(source_onto=encoder_output[0], target_onto=encoder_output[1])
        dataloader = DataLoader(llm_dataset, batch_size=batch_size, shuffle=False, collate_fn=llm_dataset.collate_fn)
        model = model_class(device=device, max_length=max_length, max_new_tokens=max_new_tokens)
        model.load(path=llm_path)
        matchings = model.generate(input_data=dataloader)
        matchings = postprocessor(matchings)
        return self._process_results(matchings, "llm", evaluate, return_matching)

    def _run_rag(self):
        return

    def _run_fewshot_rag(self):
        return

    def _run_icv_rag(self):
        return

    def _process_results(self, matchings, method, evaluate, return_matching):
        """
        Processes and evaluates the matching results.

        Parameters:
            matchings (list): List of matching results.
            method (str): The method used for alignment.
            evaluate (bool): Whether to evaluate the results.
            return_matching (bool): Whether to return the matching results.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        output_dir = self.output_path / method
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.output_format == "xml":
            xml_str = xmlify.xml_alignment_generator(matchings=matchings)
            with open(output_dir / "matchings.xml", "w", encoding="utf-8") as xml_file:
                xml_file.write(xml_str)
        elif self.output_format == "json":
            with open(output_dir / "matchings.json", "w", encoding="utf-8") as json_file:
                json.dump(matchings, json_file, indent=4, ensure_ascii=False)
        else:
            raise ValueError("Unsupported output format")

        evaluation = None
        if evaluate:
            evaluation = metrics.evaluation_report(predicts=matchings, references=self.dataset['reference'])
            print(f"{method.capitalize()} Evaluation Report:", json.dumps(evaluation, indent=4))

        if return_matching:
            return matchings

        return evaluation

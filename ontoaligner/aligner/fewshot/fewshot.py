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
This script defines the FewShotRAG class, an extension of the RAG model, designed for few-shot learning tasks.
The FewShotRAG class uses retrieval-augmented generation techniques, combining information retrieval and
language generation for enhanced predictive performance, particularly in bioinformatics and entity-matching tasks.

Classes:
- FewShotRAG: Extends the RAG class to support few-shot learning with a specific ratio of positive to negative examples.

"""
from ..rag import RAG
from .dataset import *  # NOQA
from ...postprocess import process
from typing import List, Any, Dict
import math
import random

random.seed(444)


class FewShotRAG(RAG):
    """
    A class for retrieval-augmented generation with few-shot learning, inheriting from the RAG base class.
    """

    def __init__(self, positive_ratio: float = 0.7, n_shots: int = 10, retriever_config=None, llm_config=None) -> None:
        """
        Initializes the FewShotRAG class with specified parameters.

        Parameters:
            **kwargs: Arbitrary keyword arguments.
            positive_ratio (float): The ratio of positive examples in the few-shot samples.
            n_shots (int): Number of shots to be used for few-shot learning, derived from input arguments.

        Returns:
            None
        """
        super().__init__(retriever_config=retriever_config, llm_config=llm_config)
        self.n_shots = n_shots
        self.positive_ratio = positive_ratio

    def __str__(self):
        """
        Provides a string representation of the FewShotRAG class.

        Returns:
            str: The name of the class, "FewShotRAG".
        """
        return "FewShotRAG"

    def generate(self, input_data: List[Dict]) -> List:
        """
        Generates model outputs by combining information retrieval, pre-processing, and few-shot generation.

        Parameters:
            input_data (List[Dict]): A list containing data and configurations for generation in a dict format with the following keys:
                - retriever-encoder
                - task-args
                - source-onto-iri2index
                - target-onto-iri2index

        Returns:
            List: A list of dictionaries containing the information retrieval outputs, LLM outputs,
                  and few-shot samples, each with their respective keys.
        """
        # IR generation
        ir_output = self.ir_generate(input_data=input_data)
        ir_output_cleaned = process.retriever_postprocessor(predicts=ir_output)
        examples = self.build_fewshots(input_data=input_data)
        input_data['examples'] = examples
        # LLm generation
        llm_predictions = self.llm_generate(input_data=input_data, ir_output=ir_output_cleaned)
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}, {"fewshot-samples": examples}]

    def build_llm_encoder(self, input_data: Any, llm_inputs: Any) -> Any:
        """
        Configures the LLM encoder with few-shot exemplars based on the specified dataset.

        Parameters:
            input_data (Any): Configuration data containing the 'llm-encoder' key and exemplar examples.
            llm_inputs (Any): Inputs required by the LLM encoder for building exemplars.

        Returns:
            Any: The configured dataset with exemplars.
        """
        dataset = eval(input_data["llm-encoder"])(data=llm_inputs)
        dataset.build_exemplars(examples=input_data['examples'])
        return dataset

    def build_fewshots(self, input_data: List) -> List:
        """
        Constructs few-shot examples, with a specified ratio of positive to negative samples, for training purposes.

        Positive examples are sourced from predefined references, while negative examples are randomly generated
        from non-matching source-target pairs.

        Parameters:
            input_data (List): A list containing task arguments, source ontology data, target ontology data,
                               and mappings to ontology indices.

        Returns:
            List: A list of dictionaries where each entry contains 'source', 'target', and 'answer' keys, with
                  answers indicating 'yes' for positive matches and 'no' for negative matches.
        """
        reference = input_data['task-args']['reference']
        positive_example_no = math.floor(self.positive_ratio * self.n_shots)
        negative_example_no = self.n_shots - positive_example_no

        positive_examples = random.sample(reference, positive_example_no)
        random_positive_examples = []
        for positive_example in positive_examples:
            source_iri, target_iri = positive_example['source'], positive_example['target']
            source = input_data['task-args']['source'][input_data['source-onto-iri2index'][source_iri]]
            target = input_data['task-args']['target'][input_data['target-onto-iri2index'][target_iri]]
            random_positive_examples.append([source, target])

        random_negative_examples = []
        negative_examples_source = random.sample(input_data['task-args']['source'], negative_example_no)
        negative_examples_target = random.sample(input_data['task-args']['target'], negative_example_no)
        for source, target in zip(negative_examples_source, negative_examples_target):
            source_iri, target_iri = source['iri'], target['iri']
            safe_to_add = True
            for ref in reference:
                if ref['source'] == source_iri and ref['target'] == target_iri:
                    safe_to_add = False
                    break
            if safe_to_add:
                random_negative_examples.append([source, target])

        fewshot_examples = [
            {'source': source, 'target': target, 'answer': answer}
            for examples, answer in [(random_positive_examples, 'yes'), (random_negative_examples, 'no')]
            for source, target in examples
        ]
        random.shuffle(fewshot_examples)
        print("No of random_positive_examples examples:", len(random_positive_examples))
        print("No of random_negative_examples examples:", len(random_negative_examples))
        return fewshot_examples

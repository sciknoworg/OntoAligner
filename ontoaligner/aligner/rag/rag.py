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
This script defines several classes that implement retrieval-augmented generation (RAG) architectures for natural language generation tasks.
The architecture integrates retrieval models (such as AdaRetrieval and BERTRetrieval) and language models (such as AutoModelForCausalLM and OpenAI)
to generate responses based on retrieved information.

Classes:
    - RAGBasedDecoderLLMArch: A class that implements a decoder-based LLM architecture with support for yes/no classification and token probabilities.
    - RAGBasedOpenAILLMArch: A class that integrates OpenAI’s language model for text generation with yes/no classification.
    - RAG: A base class that combines retrieval and generation models to perform RAG. It handles loading models, generating retrieval outputs,
           and creating language model inputs.
    - AutoModelDecoderRAGLLM: A subclass of RAGBasedDecoderLLMArch that uses AutoTokenizer and AutoModelForCausalLM for tokenization
      and model generation.
    - AutoModelDecoderRAGLLMV2: A subclass of RAGBasedDecoderLLMArch that uses AutoTokenizer and AutoModelForCausalLM with updated methods
      for token prediction and answer set checking.
    - OpenAIRAGLLM: A subclass of RAGBasedOpenAILLMArch designed to work with OpenAI's LLMs.
    - MambaSSMRAGLLM: A subclass of AutoModelDecoderRAGLLMV2 for MambaSSM-based generation with model loading and generation capabilities.
"""

from typing import List, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base import BaseOMModel
from ..llm import DecoderLLMArch, OpenAILLMArch
from .dataset import * # NOQA
from ...postprocess import process


class RAGBasedDecoderLLMArch(DecoderLLMArch):
    """
    RAGBasedDecoderLLMArch is a class implementing a retrieval-augmented decoder architecture.
    It generates yes/no responses using a language model, augmented by a predefined set of possible answers
    and answer sets for "yes" and "no".

    Attributes:
        ANSWER_SET (dict): A dictionary containing sets of possible answers for "yes" and "no".
        index2label (dict): Mapping from index to string label (yes/no).
        label2index (list): List of token IDs for "yes" and "no".
        answer_sets_token_id (dict): Mapping of token IDs for each answer set.
    """



    def __init__(self, **kwargs) -> None:
        """
        Initializes the RAGBasedDecoderLLMArch model.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        if "answer_set" in kwargs:
            self.ANSWER_SET = kwargs["answer_set"]
        else:
            self.ANSWER_SET = {
                "yes": ["yes", "correct", "true", "positive", "valid", "right", "accurate", "ok"],
                "no": ["no", "incorrect", "false", "negative", "invalid", "wrong", "not"],
            }
        self.index2label = {0: "yes", 1: "no"}

    def load(self, path: str) -> None:
        super().load(path=path)
        self.label2index = [self.tokenizer("yes").input_ids[-1], self.tokenizer("no").input_ids[-1]]
        self.answer_sets_token_id = {}
        for label, answer_set in self.ANSWER_SET.items():
            self.answer_sets_token_id[label] = []
            for answer in answer_set:
                if self.check_answer_set_tokenizer(answer):
                    self.answer_sets_token_id[label].append(self.tokenizer(answer).input_ids[-1])

    def __str__(self):
        """
        Returns a string representation of the RAGBasedDecoderLLMArch.

        Returns:
            str: "RAGBasedDecoderLLMArch".
        """
        return "RAGBasedDecoderLLMArch"

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        """
        Checks if the tokenizer for the given answer generates exactly two tokens.

        Args:
            answer (str): The answer to check.

        Returns:
            bool: True if the tokenizer generates exactly two tokens, False otherwise.
        """
        return len(self.tokenizer(answer).input_ids) == 2

    def get_probas_yes_no(self, outputs):
        """
        Extracts and calculates the probabilities for the "yes" and "no" responses.

        Args:
            outputs: Model outputs containing scores for the answers.

        Returns:
            torch.Tensor: A tensor containing the probabilities for "yes" and "no".
        """
        probas_yes_no = outputs.scores[0][:,
                        self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]].softmax(-1)
        return probas_yes_no

    def generate_for_llm(self, tokenized_input_data: Any) -> Any:
        """
        Generates model output based on the tokenized input data.

        Args:
            tokenized_input_data (Any): Tokenized input data for generation.

        Returns:
            outputs: The outputs from the model's generate function.
        """
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized_input_data,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.kwargs["max_new_tokens"],
                output_scores=True,
                return_dict_in_generate=True
            )
        return outputs

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        """
        Generates a prediction (yes/no) for a single input, along with its probability.

        Args:
            tokenized_input_data (Any): Tokenized input data for generation.

        Returns:
            list: A list containing the predicted sequences ("yes" or "no") and their probabilities.
        """
        outputs = self.generate_for_llm(tokenized_input_data=tokenized_input_data)
        probas_yes_no = self.get_probas_yes_no(outputs=outputs)
        yes_probas = probas_yes_no[:, : len(self.ANSWER_SET["yes"])].sum(dim=1)
        no_proba = probas_yes_no[:, len(self.ANSWER_SET["yes"]) :].sum(dim=1)
        probas = torch.cat((yes_probas.reshape(-1, 1), no_proba.reshape(-1, 1)), -1)
        probas_per_candidate_tokens = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_candidate_tokens.values]
        sequences = [
            self.index2label[int(indice)]
            for indice in probas_per_candidate_tokens.indices
        ]
        return [sequences, sequence_probas]

    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        """
        Generates predictions for multiple inputs.

        Args:
            tokenized_input_data (Any): Tokenized input data for generation.

        Returns:
            list: A list containing the predicted sequences and their probabilities.
        """
        return self.generate_for_one_input(tokenized_input_data=tokenized_input_data)


class RAGBasedOpenAILLMArch(OpenAILLMArch):
    """
    RAGBasedOpenAILLMArch is a class implementing an OpenAI-specific architecture for a RAG-based model
    with a post-processing step to extract yes/no predictions from the generated text.
    """

    def __str__(self):
        """
        Returns a string representation of the RAGBasedOpenAILLMArch.

        Returns:
            str: "RAGBasedOpenAILLMArch".
        """
        return "RAGBasedOpenAILLMArch"

    def post_processor(self, generated_texts: List) -> List:
        """
        Processes the generated texts from the OpenAI model to extract yes/no answers.

        Args:
            generated_texts (List): List of generated texts from the OpenAI model.

        Returns:
            list: A list containing the sequences ("yes" or "no") and their probabilities.
        """
        sequences, sequence_probas = [], []
        for generated_text in generated_texts:
            processed_output = generated_text.choices[0].message.content.lower()
            proba = 1
            if "yes" in processed_output:
                processed_output = "yes"
            else:
                processed_output = "no"
            sequences.append(processed_output)
            sequence_probas.append(proba)
        return [sequences, sequence_probas]


class RAG(BaseOMModel):
    """
    RAG is a retrieval-augmented generation (RAG) model that integrates both retrieval and generation components
    to answer questions based on retrieved documents and a language model.
    """

    path: str = "NO MODEL LOADING IN RAG MODELS"
    Retrieval = None
    LLM = None

    def __init__(self, retriever_config=None, llm_config=None) -> None:
        """
        Initializes the RAG model by loading the retriever and LLM components.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent class.
        """
        kwargs = {"retriever-config":retriever_config, "llm-config": llm_config}
        super().__init__(**kwargs)

        self.Retrieval = self.Retrieval(**self.kwargs["retriever-config"])
        self.LLM = self.LLM(**self.kwargs["llm-config"])

    def load(self, llm_path: str, ir_path: str) -> None:
        """
        Loads the pre-trained models for retrieval and language model generation.

        Args:
            llm_path (str): The path to the pre-trained LLM.
            ir_path (str): The path to the pre-trained retrieval model.
        """
        self.LLM.load(path=llm_path)
        self.Retrieval.load(path=ir_path)

    def __str__(self) -> str:
        """
        Returns a string representation of the RAG model.

        Returns:
            str: "RAG".
        """
        return "RAG"

    def generate(self, input_data: List) -> List:
        """
        Generates outputs using both retrieval and LLM generation components.

        Args:
            input_data (list): Input data containing retrieval encoder and task arguments.
                {
                    "retriever-encoder": self.retrieval_encoder,
                    "task-args": kwargs,
                    "source-onto-iri2index": source_onto_iri2index,
                    "target-onto-iri2index": target_onto_iri2index
                }

        Returns:
            list: A list containing the retrieval outputs and the LLM-generated outputs.
        """
        # IR generation
        ir_output = self.ir_generate(input_data=input_data)
        if 'threshold' in self.kwargs['retriever-config']:
            threshold = self.kwargs['retriever-config']['threshold']
        else:
            threshold = 0.0
        ir_output_cleaned = process.retriever_postprocessor(predicts=ir_output, threshold=threshold)
        # LLm generation
        llm_predictions = self.llm_generate(input_data=input_data, ir_output=ir_output_cleaned)
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}]

    def build_llm_inputs(self, input_data: Any, ir_output: Any) -> List:
        """
        Builds the inputs for the language model based on the retrieved outputs.

        Args:
            input_data (any): The input data containing the query.
            ir_output (any): The output from the retrieval system.

        Returns:
            list: list of inputs for the LLM.
        """
        source_onto_iri2index, target_onto_iri2index = (
            input_data["source-onto-iri2index"],
            input_data["target-onto-iri2index"],
        )
        source_onto, target_onto = (
            input_data["task-args"]["source"],
            input_data["task-args"]["target"],
        )
        llm_inputs = []
        for retrieved_items in ir_output:
            llm_inputs.append(
                {
                    "source": source_onto[
                        source_onto_iri2index[retrieved_items["source"]]
                    ],
                    "target": target_onto[
                        target_onto_iri2index[retrieved_items["target"]]
                    ],
                    "ir-scores": retrieved_items["score"],
                }
            )
        return llm_inputs

    def build_llm_encoder(self, input_data: Any, llm_inputs: Any) -> Any:
        """
        Encodes the inputs for the language model.

        Args:
            input_data (any): The input data containing the query.
            llm_inputs (any): The formatted inputs for the LLM.

        Returns:
            any: The encoded inputs for the LLM.
        """
        dataset = eval(input_data["llm-encoder"])(data=llm_inputs)
        return dataset

    def llm_generate(self, input_data: Any, ir_output: Any) -> List:
        """
        Generates predictions using the language model.

        Args:
            input_data (any): The input data containing the query.
            ir_output (any): The retrieved outputs.

        Returns:
            list: The outputs generated by the LLM.
        """
        llm_inputs = self.build_llm_inputs(input_data=input_data, ir_output=ir_output)
        dataset = self.build_llm_encoder(input_data=input_data, llm_inputs=llm_inputs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.kwargs["llm-config"]["batch_size"],
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        predictions = []
        for batch in tqdm(dataloader):
            texts, iris = batch["prompts"], batch["iris"]
            sequences, sequence_probas = self.LLM.generate(texts)
            for label, proba, iri_pair in zip(sequences, sequence_probas, iris):
                if label == "yes":
                    predictions.append({"source": iri_pair[0], "target": iri_pair[1], "score": proba})
        return predictions

    def ir_generate(self, input_data: Any) -> Any:
        """
        Generates retrieval outputs based on the input data.

        Args:
            input_data (Any): The input data containing the query.
                {
                    "retriever-encoder": self.retrieval_encoder,
                    "llm-encoder": self.llm_encoder,
                    "task-args": kwargs,
                    "source-onto-iri2index": source_onto_iri2index,
                    "target-onto-iri2index": target_onto_iri2index
                }

        Returns:
            any: The retrieval outputs.
        """
        retrieval_input = input_data["retriever-encoder"]()(**input_data["task-args"])
        retrieval_predicts = self.Retrieval.generate(input_data=retrieval_input)
        return retrieval_predicts

class AutoModelDecoderRAGLLM(RAGBasedDecoderLLMArch):
    """
    AutoModelDecoderRAGLLM is a subclass of RAGBasedDecoderLLMArch.
    It uses the AutoTokenizer and AutoModelForCausalLM models for language generation.
    """
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        """
        Returns a string representation of the AutoModelDecoderRAGLLM.

        Returns:
            str: "RAGBasedDecoderLLMArch-AutoModel".
        """
        return super().__str__() + "-AutoModel"


class AutoModelDecoderRAGLLMV2(RAGBasedDecoderLLMArch):
    """
    AutoModelDecoderRAGLLMV2 is an updated version of AutoModelDecoderRAGLLM.
    It includes additional checks for token probability predictions and optimizes answer prediction accuracy.
    """
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        """
        Returns a string representation of the AutoModelDecoderRAGLLMV2.

        Returns:
            str: "RAGBasedDecoderLLMArch-AutoModelV2".
        """
        return super().__str__() + "-AutoModelV2"

    def get_probas_yes_no(self, outputs):
        """
        Retrieves the probabilities for the "yes" and "no" labels from model output.

        Args:
            outputs: Model output containing score distributions.

        Returns:
            torch.Tensor: A tensor of probability values for "yes" and "no".
        """
        probas_yes_no = (
            outputs.scores[0][:, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]]
            .float()
            .softmax(-1)
        )
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        """
        Checks if the tokenizer produces a single token for a given answer string.

        Args:
            answer (str): The answer to validate.

        Returns:
            bool: True if only one token is generated; False otherwise.
        """
        return len(self.tokenizer(answer).input_ids) == 1


class OpenAIRAGLLM(RAGBasedOpenAILLMArch):
    """
    OpenAIRAGLLM is a subclass of RAGBasedOpenAILLMArch designed to work with OpenAI's language models.
    """
    def __str__(self):
        """
        Returns a string representation of the OpenAIRAGLLM.

        Returns:
            str: "RAGBasedOpenAILLMArch-OpenAILLM".
        """
        return super().__str__() + "-OpenAILLM"

class MambaSSMRAGLLM(AutoModelDecoderRAGLLMV2):
    """
    MambaSSMRAGLLM is a subclass of AutoModelDecoderRAGLLMV2 with support for MambaSSM,
    a model that uses efficient loading and precision settings for faster generation on compatible GPUs.
    """
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        """
        Returns a string representation of the MambaSSMRAGLLM.

        Returns:
            str: "RAGBasedDecoderLLMArch-AutoModelV2-MambaSSM".
        """
        return super().__str__() + "-MambaSSM"

    def load_model(self, path: str) -> None:
        """
        Loads the MambaSSM model with support for 8-bit precision if GPU is available.

        Args:
            path (str): Path to the model file.
        """
        if self.kwargs["device"] != "cpu":
            self.model = self.model.from_pretrained(path, load_in_8bit=True, device_map="balanced", trust_remote_code=True)
        else:
            self.model = self.model.from_pretrained(path, trust_remote_code=True)
            self.model.to(self.kwargs["device"])

    def generate_for_llm(self, tokenized_input_data: Any) -> Any:
        """
        Generates text responses using mixed precision for optimized GPU performance.

        Args:
            tokenized_input_data (Any): Tokenized input data for generation.

        Returns:
            outputs: The generated output text and probabilities.
        """
        with torch.cuda.amp.autocast():
            outputs = self.model.generate(
                tokenized_input_data.input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.kwargs["max_token_length"],
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
        return outputs

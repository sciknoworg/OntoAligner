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
    - OpenAIRAGLLM: A subclass of RAGBasedOpenAILLMArch designed to work with OpenAI's LLMs.
    - MambaSSMRAGLLM: A subclass of AutoModelDecoderRAGLLM for MambaSSM-based generation with model loading and generation capabilities.
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

    def load(self, path: str):
        super().load(path)
        self.answer_sets_token_id = {"yes": [], "no": []}

        for label, answers in self.ANSWER_SET.items():
            for answer in answers:
                token_ids = self.tokenizer(" " + answer, add_special_tokens=True).input_ids
                token_ids = self.clean_tokens(token_ids)
                if len(token_ids) > 0:
                    self.answer_sets_token_id[label].append(token_ids)

        assert len(self.answer_sets_token_id["yes"])
        assert len(self.answer_sets_token_id["no"])

    def clean_tokens(self, token_ids):
        special_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id
        }

        return [
            t for t in token_ids
            if t is not None and t not in special_tokens
        ]

    def __str__(self):
        """
        Returns a string representation of the RAGBasedDecoderLLMArch.

        Returns:
            str: "RAGBasedDecoderLLMArch".
        """
        return "RAGBasedDecoderLLMArch"

    def get_answer_probability(self, outputs, answer_tokens):

        log_prob = 0

        for i, token_id in enumerate(answer_tokens):
            logits = outputs.scores[i]

            token_log_prob = torch.log_softmax(
                logits,
                dim=-1
            )[:, token_id]

            log_prob += token_log_prob

        return log_prob / len(answer_tokens)

    def get_probas_yes_no(self, outputs):
        yes_scores = []
        for answer in self.answer_sets_token_id["yes"]:
            yes_scores.append(self.get_answer_probability(outputs, answer))

        no_scores = []
        for answer in self.answer_sets_token_id["no"]:
            no_scores.append(self.get_answer_probability(outputs, answer))

        yes_score = torch.logsumexp(torch.stack(yes_scores), dim=0)
        no_score = torch.logsumexp(torch.stack(no_scores), dim=0)
        return torch.stack([yes_score, no_score], dim=-1).softmax(-1)

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
        probas = self.get_probas_yes_no(outputs)
        values, indices = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in values]
        sequences = [self.index2label[int(indice)] for indice in indices]
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
            processed_output = generated_text.lower()
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
    Retrieval = None
    LLM = None

    def __init__(self, retriever = None, llm = None, retriever_config = None, llm_config = None) -> None:
        """
        Initializes the RAG model by loading the retriever and LLM components.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the parent class.
        """
        kwargs = {"retriever_config": retriever_config, "llm_config": llm_config}
        super().__init__(**kwargs)
        if not retriever:
            try:
                self.Retrieval = self.Retrieval(**self.kwargs["retriever_config"])
            except Exception as error:
                raise ValueError(f"{error}\n Retriever model must be provided.")
        else:
            self.Retrieval = retriever(**self.kwargs["retriever_config"])
        if not llm:
            try:
                self.LLM = self.LLM(**self.kwargs["llm_config"])
            except Exception as error:
                raise ValueError(f"{error}\n LLM model must be provided.")
        else:
            self.LLM = llm(**self.kwargs["llm_config"])

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
        if 'threshold' in self.kwargs['retriever_config']:
            threshold = self.kwargs['retriever_config']['threshold']
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
            batch_size=self.kwargs["llm_config"]["batch_size"],
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


class MambaSSMRAGLLM(AutoModelDecoderRAGLLM):
    """
    MambaSSMRAGLLM is a subclass of AutoModelDecoderRAGLLM with support for MambaSSM,
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

    def generate_for_llm(self, tokenized_input_data: Any) -> Any:
        """
        Generates text responses using mixed precision for optimized GPU performance.

        Args:
            tokenized_input_data (Any): Tokenized input data for generation.

        Returns:
            outputs: The generated output text and probabilities.
        """
        with torch.amp.autocast('cpu' if self.kwargs["device"] != "cpu" else 'cuda'):
            outputs = self.model.generate(
                tokenized_input_data.input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.kwargs["max_token_length"],
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
        return outputs

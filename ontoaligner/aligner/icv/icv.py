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
Script for implementing ICV-based ontology matching using RAG and LLM architectures.

This script applies adapter layers and ICVs to pre-trained language models, enabling few-shot
learning for ontology matching tasks. It includes modules for tokenizing demonstrations, creating
adapter layers, and fine-tuning the language model based on inference confidence vectors (ICVs).

Classes:
- AdapterLayer: Defines an adapter layer for embedding adjustment based on ICVs.
- ICVAdapter: A wrapper that integrates or removes ICV-based adapters in a model.
- ICVBasedDecoderLLMArch: A RAG-based LLM architecture for ICV-based decoding.
- ICV: Main class to manage ICV creation, ontology matching, and output generation.
- AutoModelDecoderICVLLM: LLM decoder using AutoModel for causal language modeling.
- AutoModelDecoderICVLLMV2: Extends AutoModelDecoderICVLLM with specific yes/no probability calculations.
"""
from typing import List

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tasks.demo import DemoProbInferenceForStyle
from ..rag import RAG, RAGBasedDecoderLLMArch
from ...postprocess import process

def tokenize_each_demonstration(tok, demonstration_list, dataset_name=None):
    """
    Tokenizes and preprocesses each example in a demonstration list for language model input.

    Parameters:
        tok : AutoTokenizer
            Tokenizer instance for encoding text.
        demonstration_list : list of tuple(str, str)
            List of text pairs to be tokenized, with each pair containing original and rewritten forms.
        dataset_name : str, optional
            The name of the dataset being used (default is None).

    Returns:
        list of tuple
            Tokenized list of pairs where each tuple contains encoded original and rewritten text.
    """
    tokenized_demonstration_list = []
    for exp_id in range(len(demonstration_list)):
        demonstration_list[exp_id] = (
            demonstration_list[exp_id][0].strip(" .").strip("."),
            demonstration_list[exp_id][1].strip(" .").strip("."),
        )
        # print(demonstration_list)
        e_original = tok(demonstration_list[exp_id][0])
        e_rewrite = tok(demonstration_list[exp_id][1])
        tokenized_demonstration_list.append((e_original, e_rewrite))
    return tokenized_demonstration_list


class AdapterLayer(torch.nn.Module):
    """
    Adapter layer that adjusts embeddings of a language model based on ICVs to improve
    representation for ontology matching.
    """

    def __init__(self, icvs, alpha):
        """
        Initializes an AdapterLayer instance with ICVs and an alpha scaling factor.

        Parameters:
            icvs : list of torch.Tensor
                List of ICVs used to influence model output embeddings.
            alpha : float
                Scaling factor to control the strength of ICV adjustments.
        """
        super(AdapterLayer, self).__init__()
        self.icvs = icvs
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
        """
        Forward pass of the adapter layer, adjusting embeddings based on ICVs.

        Parameters:
            x : torch.Tensor
                Input tensor representing embeddings to be transformed.

        Returns:
            torch.Tensor
                Transformed tensor with adjusted embeddings based on ICVs.
        """
        input_dtype = x.dtype
        if self.icvs is not None:
            norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)
            alpha = self.alpha
            icv_all_tasks = 0
            for i in range(len(self.icvs)):
                lambda_sim = 1.0 + torch.max(
                    torch.tensor([0.0]).to(x.device),
                    F.cosine_similarity(x.float(), self.icvs[i][None, None, :], dim=-1),
                ).unsqueeze(-1)
                icv_all_tasks -= (
                    alpha[i]
                    * lambda_sim
                    * F.normalize(self.icvs[i], dim=-1).repeat(1, x.shape[1], 1)
                )
            icv_all_tasks = 0.1 * icv_all_tasks / len(self.icvs)
            x = (
                F.normalize(F.normalize(x.float(), dim=-1) + icv_all_tasks, dim=-1)
                * norm
            )
            return x.type(input_dtype)
        else:
            return x

class ICVAdapter(torch.nn.Module):
    """
    Wrapper for integrating or removing ICV-based adjustments to a language model using adapter layers.
    """

    def __init__(self, model):
        """
        Initializes an ICVAdapter by wrapping a pre-trained model and freezing its parameters.

        Parameters:
            model : torch.nn.Module
                The pre-trained language model to be wrapped with ICV-based adapter layers.
        """
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, icvs, alpha):
        """
        Adds ICV-based adapter layers to the model for ICV-based embedding adjustment.

        Parameters:
            icvs : list of torch.Tensor
                List of ICVs used for embedding adjustment.
            alpha : list of float
                List of scaling factors for ICV influence.

        Returns:
            torch.nn.Module
                The model with ICV-based adapter layers integrated.
        """
        for i in range(0, len(self.model.transformer.h)):
            icvs_ = icvs[i]
            self.model.transformer.h[i].mlp = torch.nn.Sequential(
                self.model.transformer.h[i].mlp, AdapterLayer(icvs_, alpha)
            )
        return self.model

    def remove_adapter(self):
        """
        Removes adapter layers from the model and restores the original architecture.

        """
        weight_all = []

        for i in range(0, len(self.model.transformer.h)):
            weight_all.append(self.model.transformer.h[i].mlp[1].weight_all)
            self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]
        return weight_all


class ICVBasedDecoderLLMArch(RAGBasedDecoderLLMArch):
    """
    RAG-based decoder architecture for ICV-based LLMs used in ontology matching tasks.
    """
    icv_dataset = "demo"
    icv_prompt_version = "default"
    icv_kv_iter = 15
    icv_step_size = 0.01
    icv_num_k_shots = 3
    icv_momentum = 0.9
    icv_alpha = 1.0
    icv_seed = 0

    def __init__(self, **kwargs):
        """
        Initializes an ICVBasedDecoderLLMArch instance
        """
        super().__init__(**kwargs)
        self.task_agent = DemoProbInferenceForStyle(
            prompt_version=self.icv_prompt_version
        )
        self.task_agent.set_seed(self.icv_seed)

    def __str__(self):
        """
        String representation of the decoder class.

        Returns:
            str
                Name of the class with model type.
        """
        return "ICVBasedDecoderLLMArch"

    def build_icv(self, examples):
        """
        Builds and applies ICVs to the model to refine ontology matching accuracy.

        Parameters:
            examples : list of tuple
            List of demonstration examples for generating ICVs.
        """
        icv_examples = self.task_agent.get_icv(
            self.model, tokenize_each_demonstration(self.tokenizer, examples)
        )
        icvs_to_shift = [icv_examples]
        updated_wrapper = ICVAdapter(self.model)
        _ = updated_wrapper.get_model(
            torch.stack(icvs_to_shift, dim=1).cuda(), alpha=[self.icv_alpha]
        )


class ICV(RAG):
    """
    Core class for managing ontology matching via ICV generation and integration with LLMs.
    """

    icv_prompts = {
        "prompt-1": """Classify if the following two concepts are the same.\n### First concept:\n{source}\n### Second concept:\n{target}\n### Answer:""",
        "prompt-2": """Classify if two concepts refer to the same real word entity. \n### First concept:{source}\n### Second concept: {target}\n### Answer:""",
        "prompt-3": """Is {source} and {target} the same? The answer which can be yes or no is""",
        "prompt-4": """The task is ontology matching. Given two concepts, the task is to classify if they are the same or not.\n### The first concept is: {source}\n ### The second concept is: {target}\n### The answer which can be yes or no is:""",
        "prompt-5": """Given two concepts decide if they match or not.\n### First concept: {source}\n### Second concept: {target}\n### Answer(yes or no):""",
        "prompt-6": """The following two concepts are match or not (answer only with yes or no).\n### First concept: {source}\n### Second concept: {target}\n### Answer:"""
    }
    icv_answer_set_dict = {
        "yes-1": "yes, it is right that both concepts are the same.",
        "yes-2": "yes, true that two concepts are referring to the same real world entity.",
        "yes-3": "yes, the answer is positive, they are the same.",
        "no-1": "no, wrong, they are not the same.",
        "no-2": "no, it is false, the two concepts are not matched.",
        "no-3": "no , the answer is negative, we can not interpret this.",
    }

    def __str__(self):
        """
        String representation of the class.

        Returns:
            str
                Name of the class with model type.
        """
        return "ICVRAG"

    def generate(self, input_data: List) -> List:
        """
        Generates IR and LLM outputs for ontology matching tasks.

        Parameters:
            input_data : dict
                A dictionary containing retrieval encoder, dataset information, and task settings.
            input_data:
                {
                    "retriever-encoder": self.retrieval_encoder,
                    "task-args": kwargs,
                    "source-onto-iri2index": source_onto_iri2index,
                    "target-onto-iri2index": target_onto_iri2index
                }
        Returns:
            list of dict
                List containing IR outputs, LLM outputs, and ICV samples.
        """
        # IR generation
        ir_output = self.ir_generate(input_data=input_data)
        ir_output_cleaned = process.retriever_postprocessor(predicts=ir_output)
        examples = self.build_icv_examples(input_data=input_data)
        self.LLM.build_icv(examples=examples)
        # LLm generation
        llm_predictions = self.llm_generate(input_data=input_data, ir_output=ir_output_cleaned)
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}, {"icv-samples": examples}]


    def build_icv_examples(self, input_data: List) -> List:
        """
        Builds positive and negative ICV examples from input data for ontology matching.

        Parameters:
            input_data : dict
                Dictionary with information about the dataset and retrieval index.

        Returns:
            list of tuple
                List of queries and their expected answers for ICV-based classification.
        """
        def minor_clean(concept):
            concept = concept.replace("_", " ")
            concept = concept.lower()
            return concept

        reference = input_data['task-args']['reference']

        random_positive_examples = []
        for ref in reference:
            try:
                source_iri, target_iri = ref['source'], ref['target']
                source = input_data['task-args']['source'][input_data['source-onto-iri2index'][source_iri]]['label']
                target = input_data['task-args']['target'][input_data['target-onto-iri2index'][target_iri]]['label']
                if minor_clean(source) != minor_clean(target):
                    random_positive_examples.append([minor_clean(source), minor_clean(target)])
            except Exception as err:
                print(f"ERROR OCCURED! {err}")
            if len(random_positive_examples) == self.LLM.icv_num_k_shots:
                break

        random_negative_examples = []
        for ref in reference:
            source_iri, target_iri = ref['source'], ref['target']
            source = input_data['task-args']['source'][input_data['source-onto-iri2index'][source_iri]]['label']
            target = input_data['task-args']['target'][input_data['target-onto-iri2index'][target_iri]]['label']
            for neg_ref in reference:
                try:
                    neg_source_iri, neg_target_iri = neg_ref['source'], neg_ref['target']
                    neg_source = input_data['task-args']['source'][input_data['source-onto-iri2index'][neg_source_iri]]['label']
                    neg_target = input_data['task-args']['target'][input_data['target-onto-iri2index'][neg_target_iri]]['label']
                    if minor_clean(neg_source) != minor_clean(source) and minor_clean(target) != minor_clean(
                            neg_target) and minor_clean(neg_source) != minor_clean(neg_target):
                        random_negative_examples.append([minor_clean(source), minor_clean(neg_target)])
                        break
                except Exception as err:
                    print(f"ERROR OCCURED! {err}")
            if len(random_negative_examples) == self.LLM.icv_num_k_shots:
                break

        icv_examples = []
        for index, positive in enumerate(random_positive_examples):
            query = self.icv_prompts[f'prompt-{str(index + 1)}'].replace("{source}", positive[0])\
                                                                .replace("{target}", positive[1])
            answer = self.icv_answer_set_dict[f'yes-{str(index + 1)}']
            icv_examples.append((query, answer))

        for index, negative in enumerate(random_negative_examples):
            query = (self.icv_prompts[f'prompt-{str(index + self.LLM.icv_num_k_shots + 1)}']
                     .replace("{source}", negative[0])
                     .replace("{target}", negative[1]))
            answer = self.icv_answer_set_dict[f'no-{str(index + 1)}']
            icv_examples.append((query, answer))
        return icv_examples


class AutoModelDecoderICVLLM(ICVBasedDecoderLLMArch):
    """
    LLM decoder using AutoModel for causal language modeling, designed for ontology matching.
    """
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        """
        String representation of the decoder class.

        Returns:
            str
                Name of the class with model type.
        """
        return super().__str__() + "-AutoModel"


class AutoModelDecoderICVLLMV2(ICVBasedDecoderLLMArch):
    """
    Extended decoder class using AutoModel with specialized methods for ontology matching.
    """
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        """
        String representation of the decoder class.

        Returns:
            str
                Name of the class with model type.
        """
        return super().__str__() + "-AutoModelV2"

    def get_probas_yes_no(self, outputs):
        """
        Calculates probabilities for yes/no responses from model outputs.

        Parameters:
            outputs : torch.Tensor
                Model output tensor for calculating yes/no response probabilities.

        Returns:
            torch.Tensor
                Probabilities for yes/no classifications.
        """
        probas_yes_no = (
            outputs.scores[0][:, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]]
            .float()
            .softmax(-1)
        )
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer):
        """
        Verifies if an answer set can be tokenized as a single token.

        Parameters:
            answer : str
                Answer text to be checked.

        Returns:
            bool
                True if answer can be tokenized into a single token, False otherwise.
        """
        return len(self.tokenizer(answer).input_ids) == 1

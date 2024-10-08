# -*- coding: utf-8 -*-

from typing import List

import torch
from torch.nn import functional as F

from .tasks.demo import DemoProbInferenceForStyle
from ..rag import RAG, RAGBasedDecoderLLMArch
from ontoaligner.postprocess import process


def tokenize_each_demonstration(tok, demonstration_list, dataset_name=None):
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
    def __init__(self, icvs, alpha):
        super(AdapterLayer, self).__init__()
        self.icvs = icvs
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
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
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, icvs, alpha):
        for i in range(0, len(self.model.transformer.h)):
            icvs_ = icvs[i]
            self.model.transformer.h[i].mlp = torch.nn.Sequential(
                self.model.transformer.h[i].mlp, AdapterLayer(icvs_, alpha)
            )
        return self.model

    def remove_adapter(self):
        weight_all = []

        for i in range(0, len(self.model.transformer.h)):
            weight_all.append(self.model.transformer.h[i].mlp[1].weight_all)
            self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]
        return weight_all


class ICVBasedDecoderLLMArch(RAGBasedDecoderLLMArch):
    icv_dataset = "demo"
    icv_prompt_version = "default"
    icv_kv_iter = 15
    icv_step_size = 0.01
    icv_num_k_shots = 3
    icv_momentum = 0.9
    icv_alpha = 1.0
    icv_seed = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_agent = DemoProbInferenceForStyle(
            prompt_version=self.icv_prompt_version
        )
        self.task_agent.set_seed(self.icv_seed)

    def __str__(self):
        return "ICVBasedDecoderLLMArch"

    def build_icv(self, examples):
        icv_examples = self.task_agent.get_icv(
            self.model, tokenize_each_demonstration(self.tokenizer, examples)
        )
        icvs_to_shift = [icv_examples]
        updated_wrapper = ICVAdapter(self.model)
        _ = updated_wrapper.get_model(
            torch.stack(icvs_to_shift, dim=1).cuda(), alpha=[self.icv_alpha]
        )


class ICV(RAG):

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
        return "ICV"

    def generate(self, input_data: List) -> List:
        """
        :param input_data:
                {
                    "retriever-encoder": self.retrieval_encoder,
                    "task-args": kwargs,
                    "source-onto-iri2index": source_onto_iri2index,
                    "target-onto-iri2index": target_onto_iri2index
                }
        :return:
        """
        # IR generation
        ir_output = self.ir_generate(input_data=input_data)
        ir_output_cleaned = process.preprocess_ir_outputs(predicts=ir_output)
        examples = self.build_icv_examples(input_data=input_data)
        self.LLM.build_icv(examples=examples)
        # LLm generation
        llm_predictions = self.llm_generate(
            input_data=input_data, ir_output=ir_output_cleaned
        )
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}, {"icv-samples": examples}]

    def build_icv_examples(self, input_data: List):
        def minor_clean(concept):
            concept = concept.replace("_", " ")
            concept = concept.lower()
            return concept

        track = input_data['task-args']['dataset-info']['track']
        if track == 'bio-ml':
            reference = input_data['task-args']['reference']['equiv']['train']
        else:
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

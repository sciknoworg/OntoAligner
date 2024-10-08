# -*- coding: utf-8 -*-
from typing import Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ontoaligner.base import BaseOMModel
from ..llm import LLaMA2DecoderLLMArch, OpenAILLMArch
from .dataset import * # NOQA
from ontoaligner.postprocess import process


class RAGBasedDecoderLLMArch(LLaMA2DecoderLLMArch):
    ANSWER_SET = {
        "yes": ["yes", "correct", "true", "positive", "valid", "right", "accurate", "ok"],
        "no": ["no", "incorrect", "false", "negative", "invalid", "wrong", "not"],
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.index2label = {0: "yes", 1: "no"}
        self.label2index = [
            self.tokenizer("yes").input_ids[-1],
            self.tokenizer("no").input_ids[-1],
        ]
        self.answer_sets_token_id = {}
        for label, answer_set in self.ANSWER_SET.items():
            self.answer_sets_token_id[label] = []
            for answer in answer_set:
                if self.check_answer_set_tokenizer(answer):
                    self.answer_sets_token_id[label].append(
                        self.tokenizer(answer).input_ids[-1]
                    )

    def __str__(self):
        return "RAGBasedDecoderLLMArch"

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        return len(self.tokenizer(answer).input_ids) == 2

    def get_probas_yes_no(self, outputs):
        probas_yes_no = outputs.scores[0][
            :, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]
        ].softmax(-1)
        return probas_yes_no

    def generate_for_llm(self, tokenized_input_data: Any) -> Any:
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized_input_data,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.kwargs["max_token_length"],
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
        return outputs

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
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
        return self.generate_for_one_input(tokenized_input_data=tokenized_input_data)


class RAGBasedOpenAILLMArch(OpenAILLMArch):
    def __str__(self):
        return "RAGBasedOpenAILLMArch"

    def post_processor(self, generated_texts: List) -> List:
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
    path: str = "NO MODEL LOADING IN RAG MODELS"
    Retrieval = None
    LLM = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Retrieval = self.Retrieval(**self.kwargs["retriever-config"])
        self.LLM = self.LLM(**self.kwargs["llm-config"])

    def __str__(self):
        return "RAG"

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
        # LLm generation
        llm_predictions = self.llm_generate(input_data=input_data, ir_output=ir_output_cleaned)
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}]

    def build_llm_inputs(self, input_data: Any, ir_output: Any) -> List:
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
        dataset = eval(input_data["llm-encoder"])(data=llm_inputs)
        return dataset

    def llm_generate(self, input_data: Any, ir_output: Any) -> List:
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
            texts, iris = batch["texts"], batch["iris"]
            sequences, sequence_probas = self.LLM.generate(texts)
            for label, proba, iri_pair in zip(sequences, sequence_probas, iris):
                if label == "yes":
                    predictions.append({"source": iri_pair[0], "target": iri_pair[1], "score": proba})
        return predictions

    def ir_generate(self, input_data: Any) -> Any:
        """
        :param input_data:
                {
                    "retriever-encoder": self.retrieval_encoder,
                    "llm-encoder": self.llm_encoder,
                    "task-args": kwargs,
                    "source-onto-iri2index": source_onto_iri2index,
                    "target-onto-iri2index": target_onto_iri2index
                }
        :return:
        """
        retrieval_input = input_data["retriever-encoder"]()(**input_data["task-args"])
        retrieval_predicts = self.Retrieval.generate(input_data=retrieval_input)
        return retrieval_predicts

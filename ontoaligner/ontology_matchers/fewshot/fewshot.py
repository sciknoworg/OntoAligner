# -*- coding: utf-8 -*-
from ..rag.rag import RAG
from ..dataset import *  # NOQA
from ontoaligner.postprocess import process
from typing import List, Any
import math
import random

random.seed(444)


class FewShot(RAG):
    positive_ratio = 0.7

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_shots = int(self.kwargs['nshots'])

    def __str__(self):
        return "FewShotRAG"

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
        examples = self.build_fewshots(input_data=input_data)
        input_data['examples'] = examples
        # LLm generation
        llm_predictions = self.llm_generate(
            input_data=input_data, ir_output=ir_output_cleaned
        )
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}, {"fewshot-samples": examples}]

    def build_llm_encoder(self, input_data: Any, llm_inputs: Any) -> Any:
        dataset = eval(input_data["llm-encoder"])(data=llm_inputs)
        dataset.build_exemplars(examples=input_data['examples'])
        return dataset

    def build_fewshots(self, input_data: List) -> List:
        track = input_data['task-args']['dataset-info']['track']
        if track == 'bio-ml':
            reference = input_data['task-args']['reference']['equiv']['train']
        else:
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

        fewshot_examples = [{'source': source, 'target': target, 'answer': 'yes'} for source, target in random_positive_examples] + \
                           [{'source': source, 'target': target, 'answer': 'no'} for source, target in random_negative_examples]
        random.shuffle(fewshot_examples)
        print("No of random_positive_examples examples:", len(random_positive_examples))
        print("No of random_negative_examples examples:", len(random_negative_examples))
        return fewshot_examples

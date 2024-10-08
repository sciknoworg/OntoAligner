# -*- coding: utf-8 -*-
from typing import Any, List

from ..rag.dataset import RAGDataset


class FewShotDataset(RAGDataset):
    exemplar_prompt: str = ""

    def build_exemplars(self, examples: List):
        pass


class ConceptFewShotDataset(FewShotDataset):
    prompt: str = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).

{exemplars}
### First concept:
{source}
### Second concept:
{target}
### Answer: """

    exemplar_prompt: str = """### First concept:
{concept1}
### Second concept:
{concept2}
### Answer: {answer}

"""

    def build_exemplars(self, examples: List):
        prompt = ""
        for example in examples:
            source = self.preprocess(example["source"]["label"])
            target = self.preprocess(example["target"]["label"])
            answer = example['answer']
            prompt += self.exemplar_prompt.replace("{concept1}", source) \
                                          .replace("{concept2}", target) \
                                          .replace("{answer}", answer)
        self.exemplar_prompt = prompt

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        return self.prompt.replace("{exemplars}", self.exemplar_prompt) \
                          .replace("{source}", source) \
                          .replace("{target}", target)


class ConceptChildrenFewShotDataset(FewShotDataset):
    pass


class ConceptParentFewShotDataset(FewShotDataset):
    pass

# -*- coding: utf-8 -*-
"""
This script defines dataset classes for few-shot learning tasks, particularly for concept comparison tasks.
These classes inherit from the RAGDataset class and extend its functionality to handle few-shot learning
by constructing prompts with exemplars to help guide the model in making decisions based on limited examples.

Classes:
- FewShotDataset: A base class for creating few-shot datasets with exemplar prompts.
- ConceptFewShotDataset: A dataset class for comparing two concepts to determine if they refer to the same entity.
- ConceptChildrenFewShotDataset: A dataset class for handling concept-child relationships.
- ConceptParentFewShotDataset: A dataset class for handling concept-parent relationships.

"""
from typing import Any, List

from ..rag.dataset import RAGDataset


class FewShotDataset(RAGDataset):
    """
    A base class for few-shot datasets that constructs exemplar prompts for model guidance.

    Attributes:
        exemplar_prompt (str): A prompt template used to create exemplars for the few-shot dataset.
    """
    exemplar_prompt: str = ""

    def build_exemplars(self, examples: List):
        """
        Builds exemplar prompts based on provided examples and stores them for future use in few-shot learning.

        Parameters:
            examples (List): A list of example data used to construct the exemplar prompt.

        Returns:
            None
        """
        pass


class ConceptFewShotDataset(FewShotDataset):
    """
    A few-shot dataset class for determining if two concepts refer to the same real-world entity.

    Attributes:
        prompt (str): A template prompt for comparing two concepts and asking the model to answer 'yes' or 'no'.
        exemplar_prompt (str): A template used for creating exemplars, including source and target concepts.
    """
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
        """
        Constructs a prompt containing exemplars from provided concept examples, filling in placeholders
        for the first and second concepts and their expected answer.

        Parameters:
            examples (List): A list of dictionaries, each containing 'source', 'target', and 'answer' for concept pairs.

        Returns:
            None
        """
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
        """
        Populates the prompt with exemplars and the current input sample data for model prediction.

        Parameters:
            input_data (Any): A dictionary containing 'source' and 'target' concept data to populate the prompt.

        Returns:
            str: The completed prompt filled with exemplars and the current sample data.
        """
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        return self.prompt.replace("{exemplars}", self.exemplar_prompt) \
                          .replace("{source}", source) \
                          .replace("{target}", target)


class ConceptChildrenFewShotDataset(FewShotDataset):
    """
    A dataset class for handling few-shot learning tasks involving concept-child relationships.

    Inherits from FewShotDataset but is specific to tasks where concept relationships are hierarchical, focusing
    on child concepts in relation to parent concepts.
    """
    pass


class ConceptParentFewShotDataset(FewShotDataset):
    """
    A dataset class for handling few-shot learning tasks involving concept-parent relationships.

    Inherits from FewShotDataset but is specific to tasks where concept relationships are hierarchical, focusing
    on parent concepts in relation to child concepts.
    """
    pass

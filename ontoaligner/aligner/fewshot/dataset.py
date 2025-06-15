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
Examples:
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
        Formats the input data into a classification prompt comparing two concepts for similarity.

        Parameters:
            input_data (Any): The data sample containing source and target concepts to compare.

        Returns:
            str: The formatted classification prompt.
        """
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        prompt = self.prompt.replace("{source}", source).\
                             replace("{target}", target).\
                             replace("{exemplars}", self.exemplar_prompt)
        return prompt


class ConceptParentFewShotDataset(FewShotDataset):
    """
    A dataset class for handling few-shot learning tasks involving concept-parent relationships.

    Inherits from FewShotDataset but is specific to tasks where concept relationships are hierarchical, focusing
    on parent concepts in relation to child concepts.
    """
    prompt: str = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
Examples:
{exemplars}
### First concept:
{source}
Parents: {source_parents}
### Second concept:
{target}
Parents: {target_parents}
### Answer: """

    exemplar_prompt: str = """### First concept:
{source}
Parents: {source_parents}
### Second concept:
{target}
Parents: {target_parents}
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
            source_parents = ", ".join([self.preprocess(parent["label"]) for parent in example["source"]["parents"]])
            target_parents = ", ".join([self.preprocess(parent["label"]) for parent in example["target"]["parents"]])
            prompt += self.exemplar_prompt.replace("{source}", source).\
                                           replace("{target}", target).\
                                           replace("{source_parents}", source_parents).\
                                           replace("{target_parents}", target_parents). \
                                           replace("{answer}", answer)
        self.exemplar_prompt = prompt

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Formats the input data into a classification prompt comparing two concepts for similarity,
        with additional context from their parent concepts.

        Parameters:
            input_data (Any): The data sample containing source and target concepts with parent information.

        Returns:
            str: The formatted classification prompt.
        """
        template = self.prompt
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        source_parents = ", ".join([self.preprocess(parent["label"]) for parent in input_data["source"]["parents"]])
        target_parents = ", ".join([self.preprocess(parent["label"]) for parent in input_data["target"]["parents"]])
        template = template.replace("{source}", source).\
                            replace("{target}", target).\
                            replace("{source_parents}", source_parents).\
                            replace("{target_parents}", target_parents). \
                            replace("{exemplars}", self.exemplar_prompt)
        return template

class ConceptChildrenFewShotDataset(FewShotDataset):
    """
    A dataset class for handling few-shot learning tasks involving concept-child relationships.

    Inherits from FewShotDataset but is specific to tasks where concept relationships are hierarchical, focusing
    on child concepts in relation to parent concepts.
    """
    prompt: str = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
Examples:
{exemplars}
### First concept:
{source}
Children: {source_children}
### Second concept:
{target}
Children: {target_children}
### Answer:"""

    exemplar_prompt: str = """### First concept:
{source}
Children: {source_children}
### Second concept:
{target}
Children: {target_children}
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
            source_parents = ", ".join([self.preprocess(parent["label"]) for parent in example["source"]["childrens"]])
            target_parents = ", ".join([self.preprocess(parent["label"]) for parent in example["target"]["childrens"]])
            prompt += self.exemplar_prompt.replace("{source}", source).\
                                           replace("{target}", target).\
                                           replace("{source_children}", source_parents).\
                                           replace("{target_children}", target_parents). \
                                           replace("{answer}", answer)
        self.exemplar_prompt = prompt

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Formats the input data into a classification prompt comparing two concepts for similarity,
        with additional context from their children concepts.

        Parameters:
            input_data (Any): The data sample containing source and target concepts with children information.

        Returns:
            str: The formatted classification prompt.
        """
        template = self.prompt
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        source_children = ", ".join([self.preprocess(children["label"]) for children in input_data["source"]["childrens"]])
        target_children = ", ".join([self.preprocess(children["label"]) for children in input_data["target"]["childrens"]])
        template = template.replace("{source}", source).\
                            replace("{target}", target).\
                            replace("{source_children}", source_children).\
                            replace("{target_children}", target_children).\
                            replace("{exemplars}", self.exemplar_prompt)
        return template

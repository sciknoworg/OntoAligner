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
This script defines a set of custom dataset classes for handling various types of data used in a real-world entity classification task.
These datasets preprocess and format the input data to create structured prompts for a classification model, with variations
on how the relationship between concepts is represented (e.g., with or without parent/children context).

Classes:
    - RAGDataset: The base class for creating datasets for real-world entity classification tasks.
    - LabelRAGDataset: A subclass of RAGDataset for creating classification tasks that compare two concepts for similarity.
    - LabelParentRAGDataset: A subclass of RAGDataset that compares two concepts, considering the parent concepts of each.
    - LabelChildrenRAGDataset: A subclass of RAGDataset that compares two concepts, considering the children concepts of each.
"""
from typing import Any, Dict

from torch.utils.data import Dataset


class RAGDataset(Dataset):
    """
    A base dataset class for handling real-world entity classification tasks. This class preprocesses data and formats it into
    a suitable structure for the model's input, including creating prompts from the given concepts.

    Attributes:
        prompt (str): The template prompt used for generating classification tasks. Default is None.
        data (Any): The raw dataset provided at initialization.
        len (int): The length of the dataset.
    """
    prompt: str = None

    def __init__(self, data: Any) -> None:
        """
        Initializes the dataset with the provided data and computes the dataset length.

        Parameters:
            data (Any): The dataset to be used for classification tasks.
        """
        self.data = data
        self.len = len(data)

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the input text by replacing underscores with spaces and converting it to lowercase.

        Parameters:
            text (str): The raw text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def __getitem__(self, index: int) -> Dict:
        """
        Retrieves the data sample at the specified index and formats it into a dictionary with text and IRIs.

        Parameters:
            index (int): The index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing the processed text and IRIs for the sample.
        """
        return {
            "prompts": self.fill_one_sample(self.data[index]),
            "iris": [self.data[index]["source"]["iri"], self.data[index]["target"]["iri"]]
        }

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Placeholder method for filling a single sample. This method should be overridden by subclasses.

        Parameters:
            input_data (Any): The data sample to format.

        Returns:
            str: The formatted sample for model input.
        """
        pass

    def collate_fn(self, batchs):
        """
        Prepares a batch of data by collecting the processed texts and IRIs.

        Parameters:
            batchs (List[Dict]): A list of dictionaries containing texts and IRIs for the batch.

        Returns:
            dict: A dictionary containing lists of texts and IRIs for the batch.
        """
        batchs_clear = {"prompts": [], "iris": []}
        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear


class ConceptRAGDataset(RAGDataset):
    """
    A subclass of RAGDataset used for real-world entity classification tasks comparing two concepts
    for similarity. It formats the input data into a classification prompt with the question of whether
    two concepts refer to the same real-world entity.

    Attributes:
        prompt (str): The template prompt used for generating the classification task.
    """
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
### Second concept:
{target}
### Answer:"""

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
        return self.prompt.replace("{source}", source).replace("{target}", target)


class ConceptParentRAGDataset(RAGDataset):
    """
    A subclass of RAGDataset used for real-world entity classification tasks comparing two concepts,
    considering the parent concepts of each.

    Attributes:
        prompt (str): The template prompt used for generating the classification task.
    """
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
Parents: {source_parents}
### Second concept:
{target}
Parents: {target_parents}
### Answer:"""

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
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_parents}", source_parents)
            .replace("{target_parents}", target_parents)
        )
        return template


class ConceptChildrenRAGDataset(RAGDataset):
    """
    A subclass of RAGDataset used for real-world entity classification tasks comparing two concepts,
    considering the children concepts of each.

    Attributes:
        prompt (str): The template prompt used for generating the classification task.
    """
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
Children: {source_children}
### Second concept:
{target}
Children: {target_children}
### Answer:"""

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
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_children}", source_children)
            .replace("{target_children}", target_children)
        )
        return template

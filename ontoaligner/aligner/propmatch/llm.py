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
This script defines custom dataset classes for property-level language model (LLM) ontology matching tasks.
These datasets preprocess source and target ontology properties and format them into structured prompts for a language model,
with variations on how much property information is included, such as labels only or full metadata with domain, range, and inverse-property context.

Classes:
    - BasePropertyLLMDataset: The base class for creating property-level LLM datasets from source and target ontology properties.
    - PropertyLLMDataset: A subclass of BasePropertyLLMDataset that creates prompts using only source and target property labels.
    - PropertyFullTextLLMDataset: A subclass of BasePropertyLLMDataset that creates prompts using labels, domain, range, and inverse-property information.
"""

from typing import Any, Dict, List

from torch.utils.data import Dataset


class BasePropertyLLMDataset(Dataset):
    """
    Base dataset class for property-level LLM ontology matching.

    This class creates all possible source-target property pairs from two ontology
    property collections. It also provides shared helper methods for text
    preprocessing, field extraction, sample formatting, and batch collation.

    Attributes:
        prompt: Prompt template used by subclasses.
        data: List of source-target property-pair dictionaries.
        len: Number of source-target property pairs in the dataset.
    """

    prompt: str = None

    def __init__(self, source_onto: Any, target_onto: Any) -> None:
        """
        Initialize the dataset from source and target ontology properties.

        Args:
            source_onto: Iterable containing source ontology property dictionaries.
            target_onto: Iterable containing target ontology property dictionaries.
        """
        self.data = []

        for source in source_onto:
            for target in target_onto:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)

    def preprocess(self, text: Any) -> str:
        """
        Normalize text before inserting it into a prompt.

        The method converts the input value to a string, replaces underscores with
        spaces, lowercases the text, and converts None values to an empty string.

        Args:
            text: Text or value to normalize.

        Returns:
            Normalized text as a string.
        """
        if text is None:
            return ""

        text = str(text)
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def join_text_list(self, value: Any) -> str:
        """
        Convert a text value or list of text values into a single string.

        Args:
            value: A text value, list of text values, or None.

        Returns:
            A single string. Lists are joined with spaces, and None is converted
            to an empty string.
        """
        if value is None:
            return ""

        if isinstance(value, list):
            return " ".join(str(v) for v in value)

        return str(value)

    def get_text(self, item: Dict, key: str) -> str:
        """
        Extract and normalize a text field from an ontology property item.

        Args:
            item: Ontology property dictionary.
            key: Dictionary key to extract.

        Returns:
            Normalized text for the requested field.
        """
        return self.preprocess(self.join_text_list(item.get(key, "")))

    def __getitem__(self, index: int) -> Dict:
        """
        Return one formatted dataset sample.

        Args:
            index: Index of the source-target property pair.

        Returns:
            Dictionary containing the generated prompt and the corresponding
            source and target property IRIs.
        """
        sample = self.data[index]

        return {
            "prompts": self.fill_one_sample(sample),
            "iris": [
                sample["source"]["iri"],
                sample["target"]["iri"],
            ],
        }

    def __len__(self) -> int:
        """
        Return the number of source-target property pairs.

        Returns:
            Dataset length.
        """
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Convert one source-target property pair into a prompt.

        Subclasses must override this method to define the specific prompt format.

        Args:
            input_data: Dictionary containing source and target property data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def collate_fn(self, batchs: List[Dict]) -> Dict:
        """
        Collate multiple dataset samples into a batch.

        Args:
            batchs: List of samples returned by __getitem__.

        Returns:
            Dictionary containing batched prompts and source-target IRI pairs.
        """
        batchs_clear = {
            "prompts": [],
            "iris": [],
        }

        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])

        return batchs_clear


class PropertyLLMDataset(BasePropertyLLMDataset):
    """
    Dataset class for label-only property-level LLM ontology matching.

    This class creates prompts that compare two ontology properties using only
    their labels and asks the model whether they represent the same semantic relation.
    """

    prompt = """Determine whether the following two ontology properties represent the same semantic relation. Respond with "yes" or "no" only.
### Property 1:
{source}
### Property 2:
{target}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Build a label-only prompt for one source-target property pair.

        Args:
            input_data: Dictionary containing source and target property dictionaries.

        Returns:
            Formatted prompt string for the property pair.
        """
        source = self.preprocess(input_data["source"].get("label", ""))
        target = self.preprocess(input_data["target"].get("label", ""))

        return (
            self.prompt
            .replace("{source}", source)
            .replace("{target}", target)
        )


class PropertyFullTextLLMDataset(BasePropertyLLMDataset):
    """
    Dataset class for metadata-rich property-level LLM ontology matching.

    This class creates prompts that compare two ontology properties using their
    labels, domain information, range information, and inverse-property labels
    when available.
    """

    prompt = """Determine whether the following two ontology properties represent the same semantic relation. Respond with "yes" or "no" only.
### Property 1:
{source}
**Domain**: {source_domain}
**Range**: {source_range}
**Inverse**: {source_inverse}

### Property 2:
{target}
**Domain**: {target_domain}
**Range**: {target_range}
**Inverse**: {target_inverse}

### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Build a metadata-rich prompt for one source-target property pair.

        Args:
            input_data: Dictionary containing source and target property dictionaries,
                including optional domain, range, and inverse-property metadata.

        Returns:
            Formatted prompt string for the property pair.
        """
        source_item = input_data["source"]
        target_item = input_data["target"]

        source = self.preprocess(source_item.get("label", ""))
        target = self.preprocess(target_item.get("label", ""))

        source_domain = self.get_text(source_item, "domain_text")
        target_domain = self.get_text(target_item, "domain_text")

        source_range = self.get_text(source_item, "range_text")
        target_range = self.get_text(target_item, "range_text")

        source_inverse = ""
        if source_item.get("inverse_of"):
            source_inverse = self.get_text(source_item, "inverse_label")

        target_inverse = ""
        if target_item.get("inverse_of"):
            target_inverse = self.get_text(target_item, "inverse_label")

        return (
            self.prompt
            .replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_domain}", source_domain)
            .replace("{target_domain}", target_domain)
            .replace("{source_range}", source_range)
            .replace("{target_range}", target_range)
            .replace("{source_inverse}", source_inverse)
            .replace("{target_inverse}", target_inverse)
        )
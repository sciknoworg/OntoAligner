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
This script defines custom dataset classes for property-level retrieval-augmented generation (RAG) ontology matching tasks.
These datasets preprocess source and target ontology properties and format them into structured prompts for a language model,
with variations on how much property information is included, such as labels only or full metadata with domain, range, and inverse-property context.

Classes:
    - BasePropertyRAGDataset: The base class for creating property-level RAG datasets from source-target property pairs.
    - PropertyRAGDataset: A subclass of BasePropertyRAGDataset that creates prompts using only source and target property labels.
    - PropertyFullTextRAGDataset: A subclass of BasePropertyRAGDataset that creates prompts using labels, domain, range, and inverse-property information.
"""

from typing import Any, Dict, List

from torch.utils.data import Dataset

class BasePropertyRAGDataset(Dataset):
    """Base dataset for property-level RAG ontology matching.

    This class prepares source-target property pairs and provides shared helper
    methods for text normalization, field extraction, batching, and indexing.
    Subclasses must implement :meth:`fill_one_sample` to convert each property
    pair into a model prompt.

    Attributes:
        prompt: Prompt template used by subclasses.
        data: List of source-target property-pair dictionaries.
        len: Number of property pairs in the dataset.
    """

    prompt: str = None

    def __init__(
        self,
        data: Any = None,
        source_onto: Any = None,
        target_onto: Any = None,
    ) -> None:
        """Initialize the dataset from explicit pairs or ontology collections.

        Args:
            data: Optional precomputed list of source-target property pairs.
            source_onto: Optional iterable of source ontology properties.
            target_onto: Optional iterable of target ontology properties.

        Raises:
            ValueError: If neither ``data`` nor both ``source_onto`` and
                ``target_onto`` are provided.
        """
        if data is not None:
            self.data = data

        elif source_onto is not None and target_onto is not None:
            self.data = []

            for source in source_onto:
                for target in target_onto:
                    self.data.append(
                        {
                            "source": source,
                            "target": target,
                        }
                    )

        else:
            raise ValueError(
                "BasePropertyRAGDataset requires either data=... or source_onto=... and target_onto=..."
            )

        self.len = len(self.data)

    def preprocess(self, text: Any) -> str:
        """Normalize text for prompt construction.

        The method converts input values to strings, replaces underscores with
        spaces, lowercases the text, and converts ``None`` values to an empty
        string.

        Args:
            text: Input text or value to normalize.

        Returns:
            A normalized string.
        """
        if text is None:
            return ""

        text = str(text)
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def join_text_list(self, value: Any) -> str:
        """Convert a scalar or list-like text field into a string.

        Args:
            value: A text value, list of text values, or ``None``.

        Returns:
            A single string representation of the value. Lists are joined with
            spaces, and ``None`` is converted to an empty string.
        """
        if value is None:
            return ""

        if isinstance(value, list):
            return " ".join(str(v) for v in value)

        return str(value)

    def get_text(self, item: Dict, key: str) -> str:
        """Extract and normalize a text field from an ontology item.

        Args:
            item: Ontology property dictionary.
            key: Field name to extract from the dictionary.

        Returns:
            The normalized field value.
        """
        return self.preprocess(self.join_text_list(item.get(key, "")))

    def __getitem__(self, index: int) -> Dict:
        """Return a single prompt-ready dataset item.

        Args:
            index: Index of the source-target property pair.

        Returns:
            A dictionary containing the generated prompt and the corresponding
            source and target IRIs.
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
        """Return the number of property pairs in the dataset.

        Returns:
            Dataset length.
        """
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        """Convert a source-target property pair into a prompt.

        Subclasses must override this method with a concrete prompt generation
        strategy.

        Args:
            input_data: Source-target property-pair dictionary.

        Raises:
            NotImplementedError: Always raised by the base implementation.
        """
        raise NotImplementedError

    def collate_fn(self, batchs: List[Dict]) -> Dict:
        """Collate dataset samples into a batch dictionary.

        Args:
            batchs: List of dataset samples returned by :meth:`__getitem__`.

        Returns:
            A dictionary containing batched prompts and source-target IRI pairs.
        """
        batchs_clear = {
            "prompts": [],
            "iris": [],
        }

        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])

        return batchs_clear


class PropertyRAGDataset(BasePropertyRAGDataset):
    """Prompt dataset using only source and target property labels.

    This dataset creates binary yes/no prompts that ask whether two ontology
    properties represent the same semantic relation using only their labels.
    """

    prompt = """Determine whether the following two ontology properties represent the same semantic relation. Respond with "yes" or "no" only.

### Property 1:
{source}

### Property 2:
{target}

### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        """Build a label-only matching prompt for one property pair.

        Args:
            input_data: Dictionary containing ``source`` and ``target`` property
                dictionaries.

        Returns:
            A formatted prompt string for the property pair.
        """
        source = self.preprocess(input_data["source"].get("label", ""))
        target = self.preprocess(input_data["target"].get("label", ""))

        return (
            self.prompt
            .replace("{source}", source)
            .replace("{target}", target)
        )


class PropertyFullTextRAGDataset(BasePropertyRAGDataset):
    """Prompt dataset using labels and additional property metadata.

    This dataset creates binary yes/no prompts using property labels together
    with domain, range, and inverse-property information when available.
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
        """Build a metadata-rich matching prompt for one property pair.

        Args:
            input_data: Dictionary containing ``source`` and ``target`` property
                dictionaries with optional domain, range, and inverse metadata.

        Returns:
            A formatted prompt string for the property pair.
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
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
from typing import Any, Dict

from torch.utils.data import Dataset

class OLaLaLLMDataset(Dataset):
    """
    A dataset for OLaLa LLM candidate verification.
    """

    prompt = (
        "Classify if two descriptions refer to the same real world entity "
        "(ontology matching).\n"
        "### Concept one: endocrine pancreas secretion ### Concept two: "
        "Pancreatic Endocrine Secretion ### Answer: yes\n"
        "### Concept one: urinary bladder urothelium ### Concept two: "
        "Transitional Epithelium ### Answer: no\n"
        "### Concept one: trigeminal V nerve ophthalmic division ### Concept two: "
        "Ophthalmic Nerve ### Answer: yes\n"
        "### Concept one: foot digit 1 phalanx ### Concept two: "
        "Foot Digit 2 Phalanx ### Answer: no\n"
        "### Concept one: large intestine ### Concept two: Colon ### Answer: no\n"
        "### Concept one: ocular refractive media ### Concept two: "
        "Refractile Media ### Answer: yes\n"
        "### Concept one: {left} ### Concept two: {right} ### Answer: "
    )

    def __init__(
        self,
        source_onto: Any,
        target_onto: Any,
        candidates: Any,
        system_prompt_template: str = "{user_prompt}",
    ) -> None:
        """
        Initializes the OLaLa LLM dataset from candidate correspondences.

        Parameters:
            source_onto (Any): The encoded source ontology.
            target_onto (Any): The encoded target ontology.
            candidates (Any): The SBERT candidate predictions.
            system_prompt_template (str): Optional prompt wrapper.
        """
        self.data = []
        self.system_prompt_template = system_prompt_template

        source_map = {source["iri"]: source for source in source_onto}
        target_map = {target["iri"]: target for target in target_onto}

        for prediction in candidates:
            source_iri = prediction["source"]
            source = source_map.get(source_iri)

            if source is None:
                continue

            for target_iri in prediction.get("target-cands", []):
                target = target_map.get(target_iri)

                if target is None:
                    continue

                source_text = str(source.get("only_label", "")).strip()
                target_text = str(target.get("only_label", "")).strip()

                self.data.append({
                    "source": source,
                    "target": target,
                    "source_text": source_text,
                    "target_text": target_text,
                    "is_valid": bool(source_text and target_text),
                })

        self.len = len(self.data)

    def preprocess(self, text: str) -> str:
        """
        Preprocesses text for OLaLa LLM prompting.

        Parameters:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        return text.strip()

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Builds one OLaLa LLM prompt.

        Parameters:
            input_data (Any): One candidate pair.

        Returns:
            str: The filled prompt.
        """
        source = self.preprocess(input_data["source_text"])
        target = self.preprocess(input_data["target_text"])

        user_prompt = (
            self.prompt
            .replace("{left}", source)
            .replace("{right}", target)
        )

        return self.system_prompt_template.replace("{user_prompt}", user_prompt)

    def __getitem__(self, index: int) -> Dict:
        """
        Returns one prompt example.

        Parameters:
            index (int): The example index.

        Returns:
            Dict: The prompt, entity IRIs, and validity flag.
        """
        return {
            "prompts": self.fill_one_sample(self.data[index]),
            "iris": [
                self.data[index]["source"]["iri"],
                self.data[index]["target"]["iri"],
            ],
            "is_valid": self.data[index]["is_valid"],
        }

    def collate_fn(self, batches):
        """
        Collates OLaLa LLM examples.

        Parameters:
            batches: The batch examples.

        Returns:
            Dict: The collated batch.
        """
        batches_clear = {"prompts": [], "iris": [], "is_valid": []}

        for batch in batches:
            batches_clear["prompts"].append(batch["prompts"])
            batches_clear["iris"].append(batch["iris"])
            batches_clear["is_valid"].append(batch["is_valid"])

        return batches_clear

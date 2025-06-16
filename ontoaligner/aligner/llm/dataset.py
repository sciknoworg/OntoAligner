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


class LLMDataset(Dataset):
    prompt: str = None

    def __init__(self, source_onto: Any, target_onto: Any) -> None:
        self.data = []
        for source in source_onto:
            for target in target_onto:
                self.data.append({
                    "source": source,
                    "target": target
                })

        self.len = len(self.data)

    def preprocess(self, text: str) -> str:
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def __getitem__(self, index: int) -> Dict:
        return {
            "prompts": self.fill_one_sample(self.data[index]),
            "iris": [self.data[index]["source"]["iri"], self.data[index]["target"]["iri"]]
        }

    def __len__(self):
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        pass

    def collate_fn(self, batchs):
        batchs_clear = {"prompts": [], "iris": []}
        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear


class ConceptLLMDataset(LLMDataset):
    prompt = """Determine whether the following two concepts refer to the same real-world entity. Respond with "yes" or "no" only.
### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)


class ConceptParentLLMDataset(LLMDataset):
    prompt = """Determine whether the following two concepts, along with their parent categories, refer to the same real-world entity. Respond with "yes" or "no" only.
### Concept 1:
{source}
**Parents**: {source_parents}
### Concept 2:
{target}
**Parents**: {target_parents}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        template = self.prompt
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        source_parents = self.preprocess(input_data["source"]["parents"])
        target_parents = self.preprocess(input_data["target"]["parents"])
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_parents}", source_parents)
            .replace("{target_parents}", target_parents)
        )
        return template


class ConceptChildrenLLMDataset(LLMDataset):
    prompt = """Determine whether the following two concepts, along with their child categories, refer to the same real-world entity. Respond with "yes" or "no" only.
### Concept 1:
{source}
**Children**: {source_children}
### Concept 2:
{target}
**Children**: {target_children}
### Your Answer:  """

    def fill_one_sample(self, input_data: Any) -> str:
        template = self.prompt
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        source_children = self.preprocess(input_data["source"]["childrens"])
        target_children = self.preprocess(input_data["target"]["childrens"])
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_children}", source_children)
            .replace("{target_children}", target_children)
        )
        return template

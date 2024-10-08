# -*- coding: utf-8 -*-
from typing import Any

from torch.utils.data import Dataset


class RAGDataset(Dataset):
    prompt: str = None

    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def preprocess(self, text: str) -> str:
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def __getitem__(self, index):
        return {
            "texts": self.fill_one_sample(self.data[index]),
            "iris": [self.data[index]["source"]["iri"], self.data[index]["target"]["iri"]]
        }

    def __len__(self):
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        pass

    def collate_fn(self, batchs):
        batchs_clear = {"texts": [], "iris": []}
        for batch in batchs:
            batchs_clear["texts"].append(batch["texts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear


class LabelRAGDataset(RAGDataset):
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
### Second concept:
{target}
### Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        return self.prompt.replace("{source}", source).replace("{target}", target)


class LabelParentRAGDataset(RAGDataset):
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
Parents: {source_parents}
### Second concept:
{target}
Parents: {target_parents}
### Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
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


class LabelChildrenRAGDataset(RAGDataset):
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
Children: {source_children}
### Second concept:
{target}
Children: {target_children}
### Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
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

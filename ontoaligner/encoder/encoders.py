# -*- coding: utf-8 -*-
from typing import Any, Dict

from ..base import BaseEncoder


class LightweightEncoder(BaseEncoder):
    def parse(self, **kwargs) -> Any:
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_ontos = []
        for source in source_onto:
            encoded_source = self.get_owl_items(owl=source)
            encoded_source["text"] = self.preprocess(encoded_source["text"])
            source_ontos.append(encoded_source)
        target_ontos = []
        for target in target_onto:
            encoded_target = self.get_owl_items(owl=target)
            encoded_target["text"] = self.preprocess(encoded_target["text"])
            target_ontos.append(encoded_target)
        return [source_ontos, target_ontos]

    def __str__(self):
        return {"LightweightEncoder": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> Any:
        pass

    def get_encoder_info(self):
        return "INPUT CONSIST OF COMBINED INFORMATION TO FUZZY STRING MATCHING"


class NaiveConvOAEIEncoder(BaseEncoder):
    prompt_template: str = """<Problem Definition>
In this task, we are given two ontologies in the form of {items_in_owl}, which consist of IRI and classes.

<Ontologies-1>
{source}

<Ontologies-2>
{target}

<Objective>
Our objective is to provide ontology mapping for the provided ontologies based on their semantic similarities.

For a class in the ontology-1, which class in ontology-2 is the best match?

List matches per line.
"""

    def parse(self, **kwargs) -> Any:
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_text = ""
        for source in source_onto:
            source_text += self.get_owl_items(owl=source)
        target_text = ""
        for target in target_onto:
            target_text += self.get_owl_items(owl=target)
        prompt_sample = self.get_prefilled_prompt()
        prompt_sample = prompt_sample.replace("{source}", source_text)
        prompt_sample = prompt_sample.replace("{target}", target_text)
        return [prompt_sample]

    def __str__(self):
        return {
            "Template": super().__str__(),
            "NaiveConvOAEIPrompting": self.items_in_owl,
        }

    def get_owl_items(self, owl: Dict) -> str:
        pass

    def get_prefilled_prompt(self) -> str:
        prompt_sample = self.prompt_template
        prompt_sample = prompt_sample.replace("{items_in_owl}", self.items_in_owl)
        return prompt_sample

    def get_encoder_info(self) -> str:
        return "PROMPT-TEMPLATE: " + self.get_prefilled_prompt()


class RAGEncoder(BaseEncoder):
    retrieval_encoder: Any = None
    llm_encoder: str = None

    def parse(self, **kwargs) -> Any:
        # self.dataset_module = kwargs["dataset-module"]
        source_onto_iri2index = {
            source["iri"]: index for index, source in enumerate(kwargs["source"])
        }
        target_onto_iri2index = {
            target["iri"]: index for index, target in enumerate(kwargs["target"])
        }
        return {
            "retriever-encoder": self.retrieval_encoder,
            "llm-encoder": self.llm_encoder,
            "task-args": kwargs,
            "source-onto-iri2index": source_onto_iri2index,
            "target-onto-iri2index": target_onto_iri2index,
        }

    def __str__(self):
        return {"RagEncoder": self.items_in_owl}

    def get_encoder_info(self) -> str:
        return "PROMPT-TEMPLATE USES:" + self.llm_encoder + " ENCODER"

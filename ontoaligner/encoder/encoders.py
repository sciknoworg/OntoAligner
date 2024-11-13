# -*- coding: utf-8 -*-
"""
This script defines three encoder classes that extend the BaseEncoder class.
Each class is designed to parse ontological data in different ways, with different encoding strategies,
including lightweight encoding, naive prompting, and retrieval-augmented generation.

Classes:
    - LightweightEncoder: A base encoder for lightweight encoding for concepts.
    - NaiveConvOAEIEncoder: A base encoder for NaiveConvOAEI encoding for concepts.
    - RAGEncoder: A base encoder for RAG encoding for concepts.
"""
from typing import Any, Dict

from ..base import BaseEncoder

class LightweightEncoder(BaseEncoder):
    """
    A lightweight encoder for parsing ontology data and preprocessing it.

    This class provides methods for parsing ontological data, applying text preprocessing,
    and formatting the data into a structure suitable for further processing.
    """
    def parse(self, **kwargs) -> Any:
        """
        Parses the source and target ontologies, applying preprocessing.

        This method extracts ontology items (IRI and label) from the source and target ontologies,
        applies text preprocessing to the labels, and returns the encoded data.

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            list: A list containing two elements, the processed source and target ontologies.
        """
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
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the class name as key and items_in_owl as value.
        """
        return {"LightweightEncoder": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Abstract method for extracting ontology data.

        This method should be implemented by subclasses to extract specific ontology data
        (e.g., IRI and label) from the provided ontology item.

        Parameters:
            owl (Dict): A dictionary representing an ontology item.

        Returns:
            Any: The extracted ontology data.
        """
        pass

    def get_encoder_info(self):
        """
        Provides information about the encoder.

        Returns:
            str: A description of the encoder's function in the overall pipeline.
        """
        return "INPUT CONSIST OF COMBINED INFORMATION TO FUZZY STRING MATCHING"


class NaiveConvOAEIEncoder(BaseEncoder):
    """
    A naive encoder for ontology alignment using prompt-based methods.

    This class creates a prompt template for ontology matching, where the source and target
    ontologies are processed to generate a formatted prompt for semantic similarity analysis.
    """
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
        """
        Processes the source and target ontologies into a prompt for ontology alignment.

        This method formats the source and target ontologies into a string representation,
        filling in a pre-defined template that includes ontology items (IRI and label).

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            list: A list containing the formatted prompt string for ontology matching.
        """
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
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the template and items_in_owl values.
        """
        return {
            "Template": super().__str__(),
            "NaiveConvOAEIPrompting": self.items_in_owl,
        }

    def get_owl_items(self, owl: Dict) -> str:
        """
        Abstract method to extract ontology data as a string.

        This method should be implemented by subclasses to extract specific ontology data
        (e.g., IRI and label) from the provided ontology item.

        Parameters:
            owl (Dict): A dictionary representing an ontology item.

        Returns:
            str: The extracted ontology data as a string.
        """
        pass

    def get_prefilled_prompt(self) -> str:
        """
        Generates a prefilled prompt using the prompt template.

        This method replaces placeholders in the prompt template with the appropriate ontology information.

        Returns:
            str: The prefilled prompt with ontology items included.
        """
        prompt_sample = self.prompt_template
        prompt_sample = prompt_sample.replace("{items_in_owl}", self.items_in_owl)
        return prompt_sample

    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder and its prompt template.

        Returns:
            str: A description of the encoder's prompt template.
        """
        return "PROMPT-TEMPLATE: " + self.get_prefilled_prompt()


class RAGEncoder(BaseEncoder):
    """
    A retrieval-augmented generation (RAG) encoder for ontology mapping.

    This class leverages retrieval-augmented generation for encoding ontology data,
    allowing for both retrieval of relevant data and generation of encoded information.
    """
    retrieval_encoder: Any = None
    llm_encoder: str = None

    def parse(self, **kwargs) -> Any:
        """
        Processes the source and target ontologies into indices for retrieval and encoding.

        This method converts the source and target ontologies into mappings of IRI to index,
        preparing them for use in a retrieval-augmented generation model.

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            dict: A dictionary with the retrieval encoder, LLM encoder, task arguments,
                  and the source and target ontology index mappings.
        """
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
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the encoder's name as key and items_in_owl as value.
        """
        return {"RagEncoder": self.items_in_owl}

    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder and its usage.

        Returns:
            str: A description of the encoder's components.
        """
        return "PROMPT-TEMPLATE USES:" + self.llm_encoder + " ENCODER"

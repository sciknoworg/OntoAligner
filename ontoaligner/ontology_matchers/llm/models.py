# -*- coding: utf-8 -*-
"""
This script defines various subclasses for different types of language models (LMs), including encoder-decoder
models, decoder-only models, and models interfacing with OpenAI's GPT. These classes inherit from
predefined abstract base classes for LLM architectures and customize them for specific architectures and models.

Classes:
    - FlanT5LEncoderDecoderLM: A subclass of EncoderDecoderLLMArch, specifically for the Flan-T5 encoder-decoder
      architecture, using T5Tokenizer and T5ForConditionalGeneration for model tokenization and generation.
    - AutoModelDecoderLLM: A subclass of DecoderLLMArch, specifically for decoder-only models using AutoTokenizer
      and AutoModelForCausalLM for model tokenization and causal language modeling.
    - GPTOpenAILLM: A subclass of OpenAILLMArch, specifically for interacting with OpenAI's GPT models, providing
      a specialized implementation for this architecture.
"""

from transformers import (AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration)

from .llm import EncoderDecoderLLMArch, DecoderLLMArch, OpenAILLMArch


class FlanT5LEncoderDecoderLM(EncoderDecoderLLMArch):
    """
    A subclass of EncoderDecoderLLMArch for the Flan-T5 encoder-decoder language model.

    This class configures the tokenizer as T5Tokenizer and the model as T5ForConditionalGeneration for text generation tasks.

    Attributes:
        tokenizer (T5Tokenizer): The tokenizer class used for encoding and decoding text.
        model (T5ForConditionalGeneration): The pre-trained T5 model used for text generation tasks.
    """

    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration

    def __str__(self):
        """
        Returns a string representation of the FlanT5LEncoderDecoderLM class.

        Returns:
            str: The string representation of the FlanT5LEncoderDecoderLM class with model information.
        """
        return super().__str__() + "-FlanT5LEncoderDecoderLM"


class AutoModelDecoderLLM(DecoderLLMArch):
    """
    A subclass of DecoderLLMArch for auto-decoder language models.

    This class uses AutoTokenizer and AutoModelForCausalLM for handling tokenization and causal language modeling tasks.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer class used for encoding text.
        model (AutoModelForCausalLM): The pre-trained model used for causal language modeling tasks.
    """

    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        """
        Returns a string representation of the AutoModelDecoderLLM class.

        Returns:
            str: The string representation of the AutoModelDecoderLLM class with model information.
        """
        return super().__str__() + "-AutoModelDecoderLLM"


class GPTOpenAILLM(OpenAILLMArch):
    """
    A subclass of OpenAILLMArch specifically for interacting with OpenAI's GPT models.

    This class interfaces with OpenAI's GPT models for text generation tasks and overrides the string representation.

    Methods:
        __str__: Returns a string representation of the GPTOpenAILLM class with model information.
    """

    def __str__(self):
        """
        Returns a string representation of the GPTOpenAILLM class.

        Returns:
            str: The string representation of the GPTOpenAILLM class with model information.
        """
        return super().__str__() + "-GPTOpenAILLM"

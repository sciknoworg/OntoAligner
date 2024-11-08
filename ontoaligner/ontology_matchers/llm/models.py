# -*- coding: utf-8 -*-
from transformers import (AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration)

from .llm import EncoderDecoderLLMArch, DecoderLLMArch, OpenAILLMArch


class FlanT5LEncoderDecoderLM(EncoderDecoderLLMArch):
    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration

    def __str__(self):
        return super().__str__() + "-FlanT5LEncoderDecoderLM"


class AutoModelDecoderLLM(DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        return super().__str__() + "-AutoModelDecoderLLM"


class GPTOpenAILLM(OpenAILLMArch):
    def __str__(self):
        return super().__str__() + "-GPTOpenAILLM"

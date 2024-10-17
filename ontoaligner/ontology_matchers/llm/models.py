# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM, T5ForConditionalGeneration, T5Tokenizer

from .llm import EncoderDecoderLLMArch, DecoderLLMArch, OpenAILLMArch


class FlanT5LEncoderDecoderLM(EncoderDecoderLLMArch):
    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration

    def __str__(self):
        return super().__str__() + "-FlanT5"


class LLaMABDecoderLLM(DecoderLLMArch):
    tokenizer = LlamaTokenizer
    model = LlamaForCausalLM

    def __str__(self):
        return super().__str__() + "-LLaMALLM"


class WizardDecoderLLM(DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = LlamaForCausalLM

    def __str__(self):
        return super().__str__() + "-WizardLM"


class MistralDecoderLLM(DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = MistralForCausalLM

    def __str__(self):
        return super().__str__() + "-MistralLM"


class GPTOpenAILLM(OpenAILLMArch):
    def __str__(self):
        return super().__str__() + "-OpenAIGPT"

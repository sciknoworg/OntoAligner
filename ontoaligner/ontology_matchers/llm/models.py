# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, T5Tokenizer, LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM, T5ForConditionalGeneration

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

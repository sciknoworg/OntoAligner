# -*- coding: utf-8 -*-
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    MistralForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from .llm import EncoderDecoderLLMArch, LLaMA2DecoderLLMArch, OpenAILLMArch


class FlanT5XXLEncoderDecoderLM(EncoderDecoderLLMArch):
    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration
    path = "google/flan-t5-xxl"

    def __str__(self):
        return super().__str__() + "-FlanT5XXL"


class FlanT5XLEncoderDecoderLM(EncoderDecoderLLMArch):
    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration
    path = "google/flan-t5-xl"

    def __str__(self):
        return super().__str__() + "-FlanT5XL"


class LLaMA7BDecoderLM(LLaMA2DecoderLLMArch):
    tokenizer = LlamaTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-7b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-7B"


class LLaMA13BDecoderLM(LLaMA2DecoderLLMArch):
    tokenizer = LlamaTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-13b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-13B"


class WizardLM13BDecoderLM(LLaMA2DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = LlamaForCausalLM
    path = "WizardLM/WizardLM-13B-V1.2"

    def __str__(self):
        return super().__str__() + "-WizardLM-13B-V1.2"


class MistralLM7BDecoderLM(LLaMA2DecoderLLMArch):
    tokenizer = AutoTokenizer
    model = MistralForCausalLM
    path = "mistralai/Mistral-7B-v0.1"

    def __str__(self):
        return super().__str__() + "-MistralLM-7B-v0.1"


class GPT4OpenAILLM(OpenAILLMArch):
    path = "gpt-4-32k-0314"

    def __str__(self):
        return super().__str__() + "-GPT-4"


class ChatGPTOpenAILLM(OpenAILLMArch):
    path = "gpt-3.5-turbo-16k-0613"

    def __str__(self):
        return super().__str__() + "-GPT-3.5"

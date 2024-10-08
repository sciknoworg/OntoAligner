# -*- coding: utf-8 -*-

from ..rag.models import LLaMA7BDecoderLM, Mistral7BDecoderLM, Falcon7BDecoderLM, Vicuna7BDecoderLM, MPT7BDecoderLM, ChatGPTOpenAILLM, Mamba3BSSMLLM
from .fewshot import FewShot

from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMA7BLLMAdaFewShot(FewShot):
    Retrieval = AdaRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BAdaRAG"


class MistralLLMAdaFewShot(FewShot):
    Retrieval = AdaRetrieval
    LLM = Mistral7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLLMAdaFewShot"


class LLaMA7BLLMBertFewShot(FewShot):
    Retrieval = BERTRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BLLMBertFewShot"


class MistralLLMBertFewShot(FewShot):
    Retrieval = BERTRetrieval
    LLM = Mistral7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLLMBertFewShot"


class ChatGPTOpenAIAdaFewShot(FewShot):
    Retrieval = AdaRetrieval
    LLM = ChatGPTOpenAILLM

    def __str__(self):
        return super().__str__() + "-ChatGPTOpenAIAdaFewShot"


class FalconLLMAdaFewShot(FewShot):
    Retrieval = AdaRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaFewShot"


class FalconLLMBertFewShot(FewShot):
    Retrieval = BERTRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMBertFewShot"


class VicunaLLMAdaFewShot(FewShot):
    Retrieval = AdaRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaFewShot"


class VicunaLLMBertFewShot(FewShot):
    Retrieval = BERTRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBertFewShot"


class MPTLLMAdaFewShot(FewShot):
    Retrieval = AdaRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaFewShot"


class MPTLLMBertFewShot(FewShot):
    Retrieval = BERTRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMBertFewShot"


class MambaLLMAdaFewShot(FewShot):
    Retrieval = AdaRetrieval
    LLM = Mamba3BSSMLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMAdaFewShot"


class MambaLLMBertFewShot(FewShot):
    Retrieval = BERTRetrieval
    LLM = Mamba3BSSMLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMBertFewShot"

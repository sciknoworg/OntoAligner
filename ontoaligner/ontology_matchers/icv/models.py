# -*- coding: utf-8 -*-

from .icv import ICV, AutoModelDecoderICVLLM, AutoModelDecoderICVLLMV2
from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMALLMAdaRetrieverICVRAG(ICV):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMAdaRetrieverICVRAG"


class LLaMALLMBERTRetrieverICVRAG(ICV):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMBERTRetrieverICVRAG"


class FalconLLMAdaRetrieverICVRAG(ICV):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaRetrieverICVRAG"


class FalconLLMBERTRetrieverICVRAG(ICV):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMBERTRetrieverICVRAG"


class VicunaLLMAdaRetrieverICVRAG(ICV):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaRetrieverICVRAG"


class VicunaLLMBERTRetrieverICVRAG(ICV):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBERTRetrieverICVRAG"


class MPTLLMAdaRetrieverICVRAG(ICV):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaRetrieverICVRAG"


class MPTLLMBERTRetrieverICVRAG(ICV):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMBERTRetrieverICVRAG"

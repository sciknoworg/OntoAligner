# -*- coding: utf-8 -*-
from .rag import RAG, LLaMADecoderRAGLLM, MistralDecoderRAGLLM, OpenAIRAGLLM, AutoModelDecoderRAGLLMV2, MambaSSMRAGLLM, AutoModelDecoderRAGLLM
from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMALLMAdaRetriever(RAG):
    Retrieval = AdaRetrieval
    LLM = LLaMADecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMAdaRetriever"

class LLaMALLMBERTRetriever(RAG):
    Retrieval = BERTRetrieval
    LLM = LLaMADecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMBERTRetriever"


class MistralLLMAdaRetriever(RAG):
    Retrieval = AdaRetrieval
    LLM = MistralDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-MistralLLMAdaRetriever"


class MistralLLMBERTRetriever(RAG):
    Retrieval = BERTRetrieval
    LLM = MistralDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-MistralLLMBERTRetriever"


class GPTOpenAILLMAdaRetriever(RAG):
    Retrieval = AdaRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMAdaRetriever"

class GPTOpenAILLMBERTRetriever(RAG):
    Retrieval = BERTRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMBERTRetriever"


class FalconLLMAdaRetriever(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaRetriever"


class FalconLLMBERTRetriever(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMBERTRetriever"





class VicunaLLMAdaRetriever(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaRetriever"


class VicunaLLMBERTRetriever(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBERTRetriever"


class MPTLLMAdaRetriever(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaRetriever"


class MPTLLMBERTRetriever(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMBERTRetriever"


class MambaLLMAdaRetriever(RAG):
    Retrieval = AdaRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMAdaRetriever"


class MambaLLMBERTRetriever(RAG):
    Retrieval = BERTRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMBERTRetriever"

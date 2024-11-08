# -*- coding: utf-8 -*-
from .rag import RAG,  OpenAIRAGLLM, AutoModelDecoderRAGLLMV2, MambaSSMRAGLLM, AutoModelDecoderRAGLLM
from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMALLMAdaRetrieverRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMAdaRetrieverRAG"

class LLaMALLMBERTRetrieverRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMBERTRetrieverRAG"


class MistralLLMAdaRetrieverRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-MistralLLMAdaRetrieverRAG"


class MistralLLMBERTRetrieverRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-MistralLLMBERTRetrieverRAG"


class GPTOpenAILLMAdaRetrieverRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMAdaRetrieverRAG"

class GPTOpenAILLMBERTRetrieverRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMBERTRetrieverRAG"


class FalconLLMAdaRetrieverRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaRetriever"


class FalconLLMBERTRetrieverRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMBERTRetrieverRAG"


class VicunaLLMAdaRetrieverRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaRetrieverRAG"


class VicunaLLMBERTRetrieverRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBERTRetrieverRAG"


class MPTLLMAdaRetrieverRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaRetrieverRAG"


class MPTLLMBERTRetrieverRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMBERTRetrieverRAG"


class MambaLLMAdaRetrieverRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMAdaRetrieverRAG"


class MambaLLMBERTRetrieverRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMBERTRetrieverRAG"

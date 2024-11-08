# -*- coding: utf-8 -*-
from .fewshot import FewShotRAG
from ..rag.models import OpenAIRAGLLM, AutoModelDecoderRAGLLM, AutoModelDecoderRAGLLMV2, MambaSSMRAGLLM
from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMALLMAdaRetrieverFSRAG(FewShotRAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMAdaRetrieverFSRAG"

class LLaMALLMBERTRetrieverFSRAG(FewShotRAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-LLaMALLMBERTRetrieverFSRAG"


class MistralLLMAdaRetrieverFSRAG(FewShotRAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-MistralLLMAdaRetrieverFSRAG"


class MistralLLMBERTRetrieverFSRAG(FewShotRAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-MistralLLMBERTRetrieverFSRAG"


class GPTOpenAILLMAdaRetrieverFSRAG(FewShotRAG):
    Retrieval = AdaRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMAdaRetrieverFSRAG"

class GPTOpenAILLMBERTRetrieverFSRAG(FewShotRAG):
    Retrieval = BERTRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMBERTRetrieverFSRAG"


class FalconLLMAdaRetrieverFSRAG(FewShotRAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaRetrieverFSRAG"


class FalconLLMBERTRetrieverFSRAG(FewShotRAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-FalconLLMBERTRetrieverFSRAG"


class VicunaLLMAdaRetrieverFSRAG(FewShotRAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaRetrieverFSRAG"


class VicunaLLMBERTRetrieverFSRAG(FewShotRAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBERTRetrieverFSRAG"


class MPTLLMAdaRetrieverFSRAG(FewShotRAG):
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaRetrieverFSRAG"


class MPTLLMBERTRetrieverFSRAG(FewShotRAG):
    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        return super().__str__() + "-MPTLLMBERTRetrieverFSRAG"


class MambaLLMAdaRetrieverFSRAG(FewShotRAG):
    Retrieval = AdaRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMAdaRetrieverFSRAG"


class MambaLLMBERTRetrieverFSRAG(FewShotRAG):
    Retrieval = BERTRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMBERTRetrieverFSRAG"

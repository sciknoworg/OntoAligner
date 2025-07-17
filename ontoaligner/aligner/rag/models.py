# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script defines a series of Retrieval-Augmented Generation (RAG) classes that combine different retrieval models
and language models (LLMs). Each class specializes in pairing a specific retrieval model (e.g., AdaRetrieval, BERTRetrieval)
with a specific language model (e.g., AutoModelDecoderRAGLLM, OpenAIRAGLLM). These classes are designed to perform
retrieval-augmented generation tasks for various configurations of models.
"""

from .rag import RAG, OpenAIRAGLLM, AutoModelDecoderRAGLLMV2, MambaSSMRAGLLM, AutoModelDecoderRAGLLM
from ..retrieval.models import AdaRetrieval, SBERTRetrieval


class LLaMALLMAdaRetrieverRAG(RAG):
    """
    LLaMALLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM
    def __str__(self):
        return super().__str__() + "-LLaMALLMAdaRetrieverRAG"


class LLaMALLMBERTRetrieverRAG(RAG):
    """
    LLaMALLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM
    def __str__(self):
        return super().__str__() + "-LLaMALLMBERTRetrieverRAG"


class MistralLLMAdaRetrieverRAG(RAG):
    """
    MistralLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM
    def __str__(self):
        return super().__str__() + "-MistralLLMAdaRetrieverRAG"


class MistralLLMBERTRetrieverRAG(RAG):
    """
    MistralLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM
    def __str__(self):
        return super().__str__() + "-MistralLLMBERTRetrieverRAG"


class GPTOpenAILLMAdaRetrieverRAG(RAG):
    """
    GPTOpenAILLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the OpenAIRAGLLM language model.
    """
    Retrieval = AdaRetrieval
    LLM = OpenAIRAGLLM
    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMAdaRetrieverRAG"


class GPTOpenAILLMBERTRetrieverRAG(RAG):
    """
    GPTOpenAILLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the OpenAIRAGLLM language model.
    """
    Retrieval = SBERTRetrieval
    LLM = OpenAIRAGLLM
    def __str__(self):
        return super().__str__() + "-GPTOpenAILLMBERTRetrieverRAG"


class FalconLLMAdaRetrieverRAG(RAG):
    """
    FalconLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2
    def __str__(self):
        return super().__str__() + "-FalconLLMAdaRetriever"


class FalconLLMBERTRetrieverRAG(RAG):
    """
    FalconLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2
    def __str__(self):
        return super().__str__() + "-FalconLLMBERTRetrieverRAG"


class VicunaLLMAdaRetrieverRAG(RAG):
    """
    VicunaLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM
    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaRetrieverRAG"


class VicunaLLMBERTRetrieverRAG(RAG):
    """
    VicunaLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM
    def __str__(self):
        return super().__str__() + "-VicunaLLMBERTRetrieverRAG"


class MPTLLMAdaRetrieverRAG(RAG):
    """
    MPTLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2
    def __str__(self):
        return super().__str__() + "-MPTLLMAdaRetrieverRAG"


class MPTLLMBERTRetrieverRAG(RAG):
    """
    MPTLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2
    def __str__(self):
        return super().__str__() + "-MPTLLMBERTRetrieverRAG"


class MambaLLMAdaRetrieverRAG(RAG):
    """
    MambaLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the MambaSSMRAGLLM language model.
    """
    Retrieval = AdaRetrieval
    LLM = MambaSSMRAGLLM
    def __str__(self):
        return super().__str__() + "-MambaLLMAdaRetrieverRAG"


class MambaLLMBERTRetrieverRAG(RAG):
    """
    MambaLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the MambaSSMRAGLLM language model.
    """
    Retrieval = SBERTRetrieval
    LLM = MambaSSMRAGLLM
    def __str__(self):
        return super().__str__() + "-MambaLLMBERTRetrieverRAG"

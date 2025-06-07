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
        """
        Returns a string representation of the LLaMALLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the LLaMALLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-LLaMALLMAdaRetrieverRAG"


class LLaMALLMBERTRetrieverRAG(RAG):
    """
    LLaMALLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """

    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the LLaMALLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the LLaMALLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-LLaMALLMBERTRetrieverRAG"


class MistralLLMAdaRetrieverRAG(RAG):
    """
    MistralLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MistralLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MistralLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-MistralLLMAdaRetrieverRAG"


class MistralLLMBERTRetrieverRAG(RAG):
    """
    MistralLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """

    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MistralLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MistralLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-MistralLLMBERTRetrieverRAG"


class GPTOpenAILLMAdaRetrieverRAG(RAG):
    """
    GPTOpenAILLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the OpenAIRAGLLM language model.
    """

    Retrieval = AdaRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        """
        Returns a string representation of the GPTOpenAILLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the GPTOpenAILLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-GPTOpenAILLMAdaRetrieverRAG"


class GPTOpenAILLMBERTRetrieverRAG(RAG):
    """
    GPTOpenAILLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the OpenAIRAGLLM language model.
    """

    Retrieval = SBERTRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        """
        Returns a string representation of the GPTOpenAILLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the GPTOpenAILLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-GPTOpenAILLMBERTRetrieverRAG"


class FalconLLMAdaRetrieverRAG(RAG):
    """
    FalconLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the FalconLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the FalconLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-FalconLLMAdaRetriever"


class FalconLLMBERTRetrieverRAG(RAG):
    """
    FalconLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """

    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the FalconLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the FalconLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-FalconLLMBERTRetrieverRAG"


class VicunaLLMAdaRetrieverRAG(RAG):
    """
    VicunaLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the VicunaLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the VicunaLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-VicunaLLMAdaRetrieverRAG"


class VicunaLLMBERTRetrieverRAG(RAG):
    """
    VicunaLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    """

    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the VicunaLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the VicunaLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-VicunaLLMBERTRetrieverRAG"


class MPTLLMAdaRetrieverRAG(RAG):
    """
    MPTLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the MPTLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MPTLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-MPTLLMAdaRetrieverRAG"


class MPTLLMBERTRetrieverRAG(RAG):
    """
    MPTLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    """

    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the MPTLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MPTLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-MPTLLMBERTRetrieverRAG"


class MambaLLMAdaRetrieverRAG(RAG):
    """
    MambaLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the MambaSSMRAGLLM language model.
    """

    Retrieval = AdaRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MambaLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MambaLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-MambaLLMAdaRetrieverRAG"


class MambaLLMBERTRetrieverRAG(RAG):
    """
    MambaLLMBERTRetrieverRAG class combines the SBERTRetrieval retrieval model with the MambaSSMRAGLLM language model.
    """

    Retrieval = SBERTRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MambaLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MambaLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-MambaLLMBERTRetrieverRAG"

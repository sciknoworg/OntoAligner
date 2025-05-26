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
This script defines various subclasses for different types of language models (LMs), including encoder-decoder
models, decoder-only models, and models interfacing with OpenAI's GPT. These classes inherit from
predefined abstract base classes for LLM architectures and customize them for specific architectures and models.
"""

from transformers import (AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration)

from .llm import EncoderDecoderLLMArch, DecoderLLMArch, OpenAILLMArch


class FlanT5LEncoderDecoderLM(EncoderDecoderLLMArch):
    """
    A subclass of EncoderDecoderLLMArch for the Flan-T5 encoder-decoder language model.
    """

    tokenizer = T5Tokenizer
    model = T5ForConditionalGeneration

    def __str__(self):
        """
        Returns a string representation of the FlanT5LEncoderDecoderLM class.

        Returns:
            str: The string representation of the FlanT5LEncoderDecoderLM class with model information.
        """
        return super().__str__() + "-FlanT5LEncoderDecoderLM"


class AutoModelDecoderLLM(DecoderLLMArch):
    """
    A subclass of DecoderLLMArch for auto-decoder language models.
    """

    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __str__(self):
        """
        Returns a string representation of the AutoModelDecoderLLM class.

        Returns:
            str: The string representation of the AutoModelDecoderLLM class with model information.
        """
        return super().__str__() + "-AutoModelDecoderLLM"


class GPTOpenAILLM(OpenAILLMArch):
    """
    A subclass of OpenAILLMArch specifically for interacting with OpenAI's GPT models.
    """

    def __str__(self):
        """
        Returns a string representation of the GPTOpenAILLM class.

        Returns:
            str: The string representation of the GPTOpenAILLM class with model information.
        """
        return super().__str__() + "-GPTOpenAILLM"

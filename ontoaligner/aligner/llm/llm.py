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
This script defines several classes for interacting with large language models (LLMs) through
various architectures, such as a generic LLM class, OpenAI-based LLMs, and encoder-decoder
architectures. It provides functionalities for tokenizing input, generating text, and loading
pretrained models from specified paths. The classes are designed to handle different LLM architectures
and their respective tokenizers.

Classes:
    - LLM: An abstract base class for LLMs, defining the general structure and methods for loading
      models, tokenizing input, and generating text.
    - BaseLLMArch: A subclass of LLM, representing a base architecture for LLMs, implementing methods
      to generate text for one or multiple inputs.
    - OpenAILLMArch: A subclass of LLM for interfacing with OpenAI's GPT models, providing specific
      methods for tokenization and text generation.
    - EncoderDecoderLLMArch: A subclass of BaseLLMArch for encoder-decoder LLM architectures.
    - DecoderLLMArch: A subclass of BaseLLMArch for decoder-only LLM architectures, with special handling
      for tokenization and model loading.
"""

import time
from abc import abstractmethod
from typing import Any, List

import torch

from ...base import BaseOMModel


class LLM(BaseOMModel):
    """
    An abstract base class for implementing large language model (LLM) architectures.

    This class provides the essential structure for loading models and tokenizers, tokenizing input data,
    and generating output from LLMs. It defines abstract methods for specific implementation in subclasses.

    Attributes:
        tokenizer (Any): The tokenizer object used to process text inputs.
        model (Any): The model object used for text generation.
    """

    tokenizer: Any = None
    model: Any = None

    def __init__(self,
                 device: str="cpu",
                 truncation: bool=True,
                 max_length: int=512,
                 max_new_tokens: int=10,
                 padding: bool=True,
                 num_beams: int=1,
                 temperature: float=1.0,
                 top_p: float=1.0,
                 sleep: int=5,
                 huggingface_access_token: str=None,
                 device_map: str='balanced',
                 openai_key: str="None",
                 **kwargs) -> None:

        super().__init__(device=device,
                         truncation=truncation,
                         max_length=max_length,
                         max_new_tokens=max_new_tokens,
                         padding=padding,
                         num_beams=num_beams,
                         temperature=temperature,
                         top_p=top_p,
                         sleep=sleep,
                         huggingface_access_token=huggingface_access_token,
                         device_map=device_map,
                         openai_key=openai_key,
                         **kwargs)

    @abstractmethod
    def __str__(self):
        """
        Returns a string representation of the LLM class.

        Returns:
            str: The string representation of the LLM class.
        """
        pass

    def load(self, path: str) -> None:
        """
        Loads the tokenizer and model from the specified path.

        Args:
            path (str): The path to the pretrained model and tokenizer.
        """
        self.load_tokenizer(path=path)
        self.load_model(path=path)

    def load_tokenizer(self, path: str) -> None:
        """
        Loads the tokenizer from the specified pretrained path.

        Args:
            path (str): The path to the pretrained tokenizer.
        """
        self.tokenizer = self.tokenizer.from_pretrained(path)

    def load_model(self, path: str) -> None:
        """
        Loads the model from the specified pretrained path and moves it to the appropriate device.

        Args:
            path (str): The path to the pretrained model.
        """
        self.model = self.model.from_pretrained(path)
        self.model.to(self.kwargs['device'])

    def tokenize(self, input_data: List) -> Any:
        """
        Tokenizes the input data.

        Args:
            input_data (List): The list of text inputs to tokenize.

        Returns:
            Any: The tokenized input data in tensor format.
        """
        inputs = self.tokenizer(
            input_data,
            return_tensors="pt",
            truncation=self.kwargs["truncation"],
            max_length=self.kwargs["max_length"],
            padding=self.kwargs["padding"],
        )
        inputs.to(self.kwargs["device"])
        return inputs

    def generate(self, input_data: List) -> List:
        """
        Generates text based on the input data.

        Args:
            input_data (List): The list of input data to generate text from.

        Returns:
            List: A list of generated texts.
        """
        tokenized_input_data = self.tokenize(input_data=input_data)
        if len(input_data) == 1:
            generated_texts = self.generate_for_one_input(
                tokenized_input_data=tokenized_input_data
            )
        else:
            generated_texts = self.generate_for_multiple_input(
                tokenized_input_data=tokenized_input_data
            )
        generated_texts = self.post_processor(generated_texts=generated_texts)
        return generated_texts

    @abstractmethod
    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        """
        Generates output for a single input.

        Args:
            tokenized_input_data (Any): The tokenized input data.

        Returns:
            List: A list of generated texts for the single input.
        """
        pass

    @abstractmethod
    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        """
        Generates output for multiple inputs.

        Args:
            tokenized_input_data (Any): The tokenized input data.

        Returns:
            List: A list of generated texts for the multiple inputs.
        """
        pass

    def post_processor(self, generated_texts: List) -> List:
        """
        Processes the generated texts.

        Args:
            generated_texts (List): The list of generated texts to process.

        Returns:
            List: The processed list of generated texts.
        """
        return generated_texts


class BaseLLMArch(LLM):
    """
    A base class for LLM architectures that supports generation of text for one or multiple inputs.
    Implements the generate methods for a single and multiple inputs, and overrides model loading.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self):
        """
        Returns a string representation of the BaseLLMArch class.

        Returns:
            str: The string representation of the architecture.
        """
        pass

    def load_model(self, path: str) -> None:
        """
        Loads the model from a specified path, using different logic for CPU and non-CPU devices.

        Args:
            path (str): The path to the pretrained model.
        """
        if self.kwargs["device"] != "cpu":
            self.model = self.model.from_pretrained(self.path, load_in_8bit=True, device_map=self.kwargs['device_map'])
        else:
            super().load_model(path=path)

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        """
        Generates output for a single input using beam search.

        Args:
            tokenized_input_data (Any): The tokenized input data.

        Returns:
            List: A list of generated texts for the single input.
        """
        with torch.no_grad():
            sequence_ids = self.model.generate(
                **tokenized_input_data,
                num_beams=self.kwargs["num_beams"],
                max_new_tokens=self.kwargs["max_new_tokens"],
                temperature=self.kwargs["temperature"],
                top_p=self.kwargs["top_p"],
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            generated_ids = sequence_ids["sequences"][:, tokenized_input_data.input_ids.shape[-1]:]

        sequences = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return sequences

    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        """
        Generates output for multiple inputs.

        Args:
            tokenized_input_data (Any): The tokenized input data.

        Returns:
            List: A list of generated texts for the multiple inputs.
        """
        with torch.no_grad():
            sequence_ids = self.model.generate(
                input_ids=tokenized_input_data["input_ids"],
                attention_mask=tokenized_input_data["attention_mask"],
                max_new_tokens=self.kwargs["max_new_tokens"],
                temperature=self.kwargs["temperature"],
                top_p=self.kwargs["top_p"],
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            generated_ids = sequence_ids["sequences"][:, tokenized_input_data.input_ids.shape[-1]:]
        sequences = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return sequences


class OpenAILLMArch(LLM):
    """
    A subclass of LLM for interfacing with OpenAI's GPT models. It uses OpenAI's API for generating text.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self):
        """
        Returns a string representation of the OpenAILLM class.

        Returns:
            str: The string representation of the OpenAILLM class.
        """
        return "OpenAILLM"

    def tokenize(self, input_data: List) -> Any:
        """
        Tokenizes the input data. For OpenAI models, it returns the input data as is.

        Args:
            input_data (List): The list of input data to tokenize.

        Returns:
            Any: The input data, unchanged for OpenAI models.
        """
        return input_data

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        """
        Generates output for a single input using OpenAI's GPT models.

        Args:
            tokenized_input_data (Any): The tokenized input data.

        Returns:
            List: A list of generated texts for the single input.
        """
        if len(tokenized_input_data[0].split(", ")) > 1000:
            print("REDUCTION of the INPUT")
            tokenized_input_data[0] = ", ".join(
                tokenized_input_data[0].split(", ")[:1000]
            )
        prompt = [{"role": "user", "content": tokenized_input_data[0]}]
        is_generated_output = False
        response = None
        while not is_generated_output:
            try:
                response = self.client.chat.completions.create(
                    model=self.path,
                    messages=prompt,
                    temperature=self.kwargs["temperature"],
                    max_tokens=self.kwargs["max_new_tokens"],
                    # top_p=self.kwargs["top_p"],
                )
                is_generated_output = True
            except Exception as error:
                print(
                    f"Unexpected {error}, {type(error)} \n"
                    f"Going for sleep for {self.kwargs['sleep']} seconds!"
                )
                time.sleep(self.kwargs["sleep"])
        return [response]

    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        """
        Generates output for multiple inputs using OpenAI's GPT models.

        Args:
            tokenized_input_data (Any): The tokenized input data.

        Returns:
            List: A list of generated texts for the multiple inputs.
        """
        responses = []
        for input_data in tokenized_input_data:
            response = self.generate_for_one_input(tokenized_input_data=[input_data])[0]
            responses.append(response)
        return responses

    def post_processor(self, generated_texts: List) -> List:
        """
        Processes the generated texts by extracting the relevant content from the response.

        Args:
            generated_texts (List): A list of generated texts from the OpenAI API.

        Returns:
            List: A list of processed outputs containing the relevant content from the generated texts.
        """
        processed_outputs = []
        for generated_text in generated_texts:
            try:
                processed_output = generated_text["choices"][0]["message"]["content"]
            except Exception:
                processed_output = generated_text.choices[0].message.content.lower()
            processed_outputs.append(processed_output)
        return processed_outputs


class EncoderDecoderLLMArch(BaseLLMArch):
    """
    Specialized LLM subclass representing encoder-decoder architectures for handling sequence-to-sequence generation tasks.
    """
    def __str__(self):
        """
        Returns a string representation of the EncoderDecoderLLMArch class.

        Returns:
            str: String representation of EncoderDecoderLLMArch class.
        """
        return "EncoderDecoderLLMArch"


class DecoderLLMArch(BaseLLMArch):
    """
    Specialized subclass of BaseLLMArch for decoder-only architectures, supporting unique tokenization
    and model-loading configurations required by specific LLM types.

    Attributes:
        llms_with_special_tk (List[str]): List of LLMs that require special tokenization handling.
        llms_with_hugging_tk (List[str]): List of LLMs that require a Hugging Face access token.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes DecoderLLMArch with specific LLM lists for special tokenization and
        Hugging Face token requirements.
        """
        super().__init__(**kwargs)
        self.llms_with_special_tk =  ["llama", "falcon", "vicuna", "mpt", 'mamba',  'qwen']
        self.llms_with_hugging_tk = ["llama", 'mistral']

    def __str__(self):
        """
        Returns a string representation of the DecoderLLMArch class.

        Returns:
            str: String representation indicating the DecoderLLMArch.
        """
        return "DecoderLLMArch"

    def check_list_llms(self, llm_path: str, check_list: List[str]) -> bool:
        """
        Checks if any LLMs in a specified list are present in the model path.

        Args:
            llm_path (str): Path to the LLM model.
            check_list (List[str]): List of LLM identifiers to check.

        Returns:
            bool: True if any item in check_list is in llm_path, False otherwise.
        """
        for llm in check_list:
            if llm in llm_path.lower():
                return True
        return False

    def load_tokenizer(self, path: str) -> None:
        """
        Loads the tokenizer with specific configurations based on the LLM type.
        Special handling is applied if the model requires a Hugging Face access token or
        specific padding configuration.

        Args:
            path (str): Path to the pretrained tokenizer.
        """
        llm_req_special_tk = self.check_list_llms(path, self.llms_with_special_tk)
        llm_req_hugging_tk = self.check_list_llms(path, self.llms_with_hugging_tk)

        if llm_req_special_tk and llm_req_hugging_tk:
            self.tokenizer = self.tokenizer.from_pretrained(path, token=self.kwargs['huggingface_access_token'], padding_side="left")
        elif llm_req_special_tk:
            self.tokenizer = self.tokenizer.from_pretrained(path, padding_side="left")
        elif llm_req_hugging_tk:
            self.tokenizer = self.tokenizer.from_pretrained(path, token=self.kwargs['huggingface_access_token'])
        else:
            self.tokenizer = self.tokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, path: str) -> None:
        """
        Loads the model with device-specific configurations and handles requirements for Hugging Face token access.
        Adjusts precision and device mapping for non-CPU devices if specified.

        Args:
            path (str): Path to the pretrained model.
        """
        llm_req_hugging_tk = self.check_list_llms(path, self.llms_with_hugging_tk)

        if self.kwargs["device"] != "cpu":
            if llm_req_hugging_tk:
                self.model = self.model.from_pretrained(path, load_in_8bit=True, device_map=self.kwargs['device_map'], token=self.kwargs['huggingface_access_token'])
            else:
                self.model = self.model.from_pretrained(path, load_in_8bit=True, device_map="balanced")
        else:
            self.model = self.model.from_pretrained(path, token=self.kwargs['huggingface_access_token'])
            if llm_req_hugging_tk:
                self.model = self.model.from_pretrained(path, token=self.kwargs['huggingface_access_token'])
            else:
                self.model = self.model.from_pretrained(path)

            self.model.to(self.kwargs["device"])

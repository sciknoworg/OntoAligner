# -*- coding: utf-8 -*-
import os
import time
from abc import abstractmethod
from typing import Any, List

import torch
from openai import OpenAI

from ontoaligner.base import BaseOMModel


class LLM(BaseOMModel):
    tokenizer: Any = None
    model: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load()

    @abstractmethod
    def __str__(self):
        pass

    def load(self) -> None:
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self) -> None:
        self.tokenizer = self.tokenizer.from_pretrained(self.path)

    def load_model(self) -> None:
        self.model = self.model.from_pretrained(self.path)
        self.model.to(self.kwargs["device"])

    def tokenize(self, input_data: List) -> Any:
        inputs = self.tokenizer(
            input_data,
            return_tensors="pt",
            truncation=self.kwargs["truncation"],
            max_length=self.kwargs["tokenizer_max_length"],
            padding=self.kwargs["padding"],
        )
        inputs.to(self.kwargs["device"])
        return inputs

    def generate(self, input_data: List) -> List:
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
        pass

    @abstractmethod
    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        pass

    def post_processor(self, generated_texts: List) -> List:
        return generated_texts


class BaseLLMArch(LLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self):
        pass

    def load_model(self) -> None:
        if self.kwargs["device"] != "cpu":
            self.model = self.model.from_pretrained(
                self.path, load_in_8bit=True, device_map="balanced"
            )
        else:
            super().load_model()

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
        with torch.no_grad():
            sequence_ids = self.model.generate(
                tokenized_input_data.input_ids,
                num_beams=self.kwargs["num_beams"],
                max_new_tokens=self.kwargs["max_token_length"],
                temperature=self.kwargs["temperature"],
                top_p=self.kwargs["top_p"],
            )
        sequences = self.tokenizer.batch_decode(sequence_ids.cpu(), skip_special_tokens=True)
        return sequences

    def generate_for_multiple_input(self, tokenized_input_data: Any) -> List:
        with torch.no_grad():
            sequence_ids = self.model.generate(
                input_ids=tokenized_input_data["input_ids"],
                attention_mask=tokenized_input_data["attention_mask"],
                max_new_tokens=self.kwargs["max_token_length"],
                temperature=self.kwargs["temperature"],
                top_p=self.kwargs["top_p"],
            )
        sequences = self.tokenizer.batch_decode(
            sequence_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return sequences


class OpenAILLMArch(LLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    def __str__(self):
        return "OpenAILM"

    def load(self) -> None:
        pass

    def tokenize(self, input_data: List) -> Any:
        return input_data

    def generate_for_one_input(self, tokenized_input_data: Any) -> List:
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
                    max_tokens=self.kwargs["max_token_length"],
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
        responses = []
        for input_data in tokenized_input_data:
            response = self.generate_for_one_input(tokenized_input_data=[input_data])[0]
            responses.append(response)
        return responses

    def post_processor(self, generated_texts: List) -> List:
        processed_outputs = []
        for generated_text in generated_texts:
            try:
                processed_output = generated_text["choices"][0]["message"]["content"]
            except Exception:
                processed_output = generated_text.choices[0].message.content.lower()
            processed_outputs.append(processed_output)
        return processed_outputs


class EncoderDecoderLLMArch(BaseLLMArch):
    def __str__(self):
        return "EncoderDecoderLLMArch"


class DecoderLLMArch(BaseLLMArch):
    def __str__(self):
        return "DecoderLLMArch"


class LLaMA2DecoderLLMArch(BaseLLMArch):
    def __str__(self):
        return "LLaMA2DecoderLLMArch"

    def load_tokenizer(self) -> None:
        def padding_side_left_llms(path):
            llms = ["llama", "falcon", "vicuna", "mpt", 'Mamba']
            for llm in llms:
                if llm in path:
                    return True
            return False

        if padding_side_left_llms(self.path):
            self.tokenizer = self.tokenizer.from_pretrained(
                self.path,
                token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
                padding_side="left",
            )
        else:
            self.tokenizer = self.tokenizer.from_pretrained(self.path, token=os.environ["HUGGINGFACE_ACCESS_TOKEN"])
        if "falcon" not in self.path:
            self.tokenizer.eos_token = "<\s>"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def load_model(self) -> None:
        if self.kwargs["device"] != "cpu":
            self.model = self.model.from_pretrained(
                self.path,
                load_in_8bit=True,
                device_map="balanced",
                token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            )
        else:
            self.model = self.model.from_pretrained(self.path, token=os.environ["HUGGINGFACE_ACCESS_TOKEN"])
            self.model.to(self.kwargs["device"])

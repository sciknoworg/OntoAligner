# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any


class BaseEncoder(ABC):
    prompt_template: str = ""
    items_in_owl: str = ""

    def __str__(self):
        return self.prompt_template

    def preprocess(self, text: str) -> str:
        text = text.replace("_", " ")
        text = text.lower()
        return text

    @abstractmethod
    def parse(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_encoder_info(self) -> str:
        pass

    def __call__(self, **kwargs):
        return self.parse(**kwargs)

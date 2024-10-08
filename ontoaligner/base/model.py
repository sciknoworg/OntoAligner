# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List


class BaseOMModel(ABC):
    path: str = ""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def generate(self, input_data: List) -> List:
        pass

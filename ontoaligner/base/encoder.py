# -*- coding: utf-8 -*-
"""
This script defines the `BaseEncoder` class, which serves as an abstract base class for creating encoders
that handle text preprocessing and encoding tasks. The class provides a method for preprocessing text,
defining abstract methods for encoding and retrieving encoder-specific information, and implementing a
call method to invoke the parsing behavior.

Classes:
    - BaseEncoder: An abstract base class for encoders, providing text preprocessing, a template for prompts,
                   and methods for encoding data and obtaining encoder information.
"""
from abc import ABC, abstractmethod
from typing import Any

class BaseEncoder(ABC):
    """
    An abstract base class for encoders that provides methods for text preprocessing and encoding tasks.
    This class defines methods for preprocessing text and serves as a blueprint for creating encoders
    that will handle specific encoding logic and retrieval of encoder-related information.

    Attributes:
        prompt_template (str): A string template used in prompting for encoding tasks.
        items_in_owl (str): A string that defines the items in the ontology used by the encoder.

    Methods:
        __str__: Returns the prompt template string.
        preprocess: Preprocesses input text by replacing underscores with spaces and converting it to lowercase.
        parse: An abstract method that should be implemented to parse input data.
        get_encoder_info: An abstract method that should be implemented to provide encoder-specific information.
        __call__: Calls the `parse` method, making the encoder instance callable for encoding tasks.
    """

    prompt_template: str = ""
    items_in_owl: str = ""

    def __str__(self):
        """
        Returns the prompt template string.

        This method simply returns the string value of the `prompt_template` attribute.

        Returns:
            str: The `prompt_template` string.
        """
        return self.prompt_template

    def preprocess(self, text: str) -> str:
        """
        Preprocesses input text by replacing underscores with spaces and converting the text to lowercase.

        This method is used to standardize the format of input text before processing it further for encoding.

        Parameters:
            text (str): The input text that needs preprocessing.

        Returns:
            str: The preprocessed text with underscores replaced by spaces and all characters in lowercase.
        """
        text = text.replace("_", " ")
        text = text.lower()
        return text

    @abstractmethod
    def parse(self, **kwargs) -> Any:
        """
        An abstract method for parsing input data. Subclasses must implement this method.

        This method is intended to be overridden by subclasses to define how to parse input data for encoding.

        Parameters:
            **kwargs: The keyword arguments passed to the method for parsing.

        Returns:
            Any: The parsed data in a format defined by the subclass.
        """
        pass

    @abstractmethod
    def get_encoder_info(self) -> str:
        """
        An abstract method for retrieving encoder-specific information. Subclasses must implement this method.

        This method is intended to be overridden by subclasses to return relevant information about the encoder.

        Returns:
            str: Information about the encoder (e.g., type, configuration).
        """
        pass

    def __call__(self, **kwargs):
        """
        Makes the encoder instance callable by calling the `parse` method.

        This method allows instances of the `BaseEncoder` (or its subclasses) to be called directly, which will
        invoke the `parse` method for encoding tasks.

        Parameters:
            **kwargs: The keyword arguments passed to the `parse` method.

        Returns:
            Any: The result of calling the `parse` method.
        """
        return self.parse(**kwargs)

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
Defines a blueprint for ontology matching models, specifying methods for string representation and data generation that must be implemented by subclasses.
The script ensures consistency and structure for building specialized models in the ontology matching domain.

Classes:
    - BaseOMModel: An abstract base class for ontology matching models, which defines methods for
                   string representation and data generation.
"""
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI


class BaseOMModel(ABC):
    """
    An abstract base class for ontology matching models. This class defines methods for string representation
    and output generation, which must be implemented by subclasses.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the ontology matching model with optional keyword arguments.

        Parameters:
            **kwargs: Additional keyword arguments that may be used for model configuration or parameters.
        """
        self.client = OpenAI(api_key=kwargs.get('openai_key', "None"))
        self.kwargs = kwargs

    @abstractmethod
    def __str__(self):
        """
        Returns a string representation of the model. This method must be implemented by subclasses.

        Returns:
            str: A string that represents the model. The specific implementation should provide meaningful
                 details about the model's configuration or state.
        """
        pass

    @abstractmethod
    def generate(self, input_data: List) -> List:
        """
        Generates output based on the input data. This method must be implemented by subclasses.

        Parameters:
            input_data (List): A list of data that will be processed by the model to generate output.

        Returns:
            List: A list containing the generated output based on the input data. The specific content of
                  the output will depend on the model's functionality.
        """
        pass

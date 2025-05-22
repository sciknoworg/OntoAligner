# -*- coding: utf-8 -*-
"""
This script defines the DemoProbInferenceForStyle class for performing probabilistic inference
with exemplar-based prompt learning in a demo context. It inherits from the BaseProbInference class
and provides specific configurations for dataset loading, prompt version handling, and exemplar management.

Classes:
    DemoProbInferenceForStyle: A demonstration-specific subclass of BaseProbInference, implementing
                               methods for dataset signature, prompt version handling, and exemplar
                               formatting for probabilistic inference.

"""

# -*- coding: utf-8 -*-
from .base import BaseProbInference


class DemoProbInferenceForStyle(BaseProbInference):
    """
    A subclass of BaseProbInference tailored for demonstration-style probabilistic inference
    with specific dataset configurations and prompt versioning.

    Attributes:
        can_be_stratified (bool): Indicates whether stratified sampling is enabled.
        num_base_shot (int): Number of base shots for exemplar selection.
    """

    def __init__(self, prompt_version):
        """
        Initializes DemoProbInferenceForStyle with a specified prompt version and sets
        exemplar parameters for base shot count and stratification.

        Args:
            prompt_version (str): The version of the prompt to be used in the inference process.
        """
        super().__init__(prompt_version)
        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        """
        Returns the default prompt version for this demonstration class.

        Returns:
            str: Default prompt version string, "sp".
        """
        return "sp"

    def dataset_signature(self):
        """
        Provides the dataset signature specifying the source and split for sample and result data.

        Returns:
            dict: Dictionary containing dataset information for "sample" and "result" keys.
                  Each key maps to a tuple with dataset name, subset, and split details.
        """
        return {
            "sample": ("demo", None, "train"),
            "result": ("demo", None, "test"),
        }

    def dataset_preprocess(self, raw_data):
        """
        Processes raw data for use in the inference model. Intended to be implemented or overridden
        by subclasses as per specific data preprocessing requirements.

        Args:
            raw_data (list): List of raw data samples for preprocessing.
        """
        pass

    def handcrafted_exemplars(self):
        """
        Placeholder for generating handcrafted exemplars. Meant to be overridden by subclasses
        to implement specific exemplar generation logic.

        Raises:
            NotImplementedError: Raised to indicate the method should be implemented in subclasses.
        """
        raise NotImplementedError

    def exemplar_seperator(self):
        """
        Provides the separator string to use between exemplars, based on the prompt version.

        Returns:
            str: Separator string for exemplars.

        Raises:
            ValueError: If an unsupported prompt version is specified.
        """
        if self.prompt_version.startswith("sp"):
            return ".  "
        else:
            raise ValueError(
                f"AGNews: Not supported prompt_version: {self.prompt_version}"
            )

    def paralell_style_promptify(self, query, return_reference=False, Instruction=""):
        """
        Formats a query as a prompt for parallel-style inference, suitable for use in demonstrations.

        Args:
            query (str): The input query to be formatted.
            return_reference (bool, optional): If True, returns a reference along with the formatted prompt.
            Instruction (str, optional): An instruction string to include in the prompt format.

        Returns:
            None: Placeholder, intended to be implemented with custom logic in subclasses.
        """
        pass

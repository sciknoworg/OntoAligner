# -*- coding: utf-8 -*-
"""
This script provides utility functions for directory creation and output file path generation.
It defines functions to create directories and generate output file paths for saving results
with a specific naming convention, including timestamping.

Functions:
- mkdir: Creates a directory if it doesn't already exist.
- make_output_dir: Generates a structured output directory and file path based on provided parameters.
"""

import os
import time
from typing import Dict


def mkdir(path: str) -> None:
    """
    Creates a directory at the specified path if it does not already exist.

    Parameters:
        path (str): The path where the directory should be created.

    Returns:
        None: This function does not return anything.
    """
    if not os.path.exists(path):
        os.mkdir(path)


def make_output_dir(
    output_dir: str, model_id: str, dataset_info: Dict, encoder_id: str, approach: str
) -> str:
    """
    Generates a structured output directory path and a timestamped filename for storing results.

    This function creates the necessary directories for a given dataset and model,
    then constructs a file path with a timestamp for saving the output.

    Parameters:
        output_dir (str): The base directory for storing outputs.
        model_id (str): The identifier of the model being used.
        dataset_info (Dict): A dictionary containing dataset information, specifically:
            - "track": The track or task name in the dataset.
            - "ontology-name": The ontology name related to the dataset.
        encoder_id (str): The identifier of the encoder used.
        approach (str): The name of the approach being used in the task.

    Returns:
        str: The generated file path string where the output should be saved.
    """
    track_output_dir = os.path.join(output_dir, dataset_info["track"])
    mkdir(track_output_dir)
    track_task_output_dir = os.path.join(
        track_output_dir, dataset_info["ontology-name"]
    )
    mkdir(track_task_output_dir)
    named_tuple = time.localtime()
    time_string = time.strftime("%Y.%m.%d-%H:%M:%S", named_tuple)
    output_file_path = os.path.join(
        track_task_output_dir,
        f"{approach}-{model_id}-{encoder_id}-{time_string}.json",
    )
    return output_file_path

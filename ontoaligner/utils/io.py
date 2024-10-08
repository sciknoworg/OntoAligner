# -*- coding: utf-8 -*-
"""
Includes Input/Output (I/O) functionalities like reading and writing from and into specific file formats.
"""
import json
import os
from typing import Any, Dict

import pandas as pd


def read_tsv(input_path: str) -> pd:
    """
        Read the `tsv` file of the given ``input_path``.
    :param input_path: Path to the tsv file
    :return: A loaded tsv file
    """
    return pd.read_csv(input_path, sep="\t")


def read_json(input_path: str) -> Dict[str, Any]:
    """
    Reads the ``json`` file of the given ``input_path``.

    :param input_path: Path to the json file
    :return: A loaded json object.
    """
    with open(input_path, encoding="utf-8") as f:
        json_data = json.load(f)

    return json_data


def write_json(output_path: str, json_data: Any):
    """
    Write the ``json_data`` to the ``output_path`` file.

    :param output_path:  Path to output json file
    :param json_data: A json Data
    :return:
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(json_data, outfile, indent=4, ensure_ascii=False)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

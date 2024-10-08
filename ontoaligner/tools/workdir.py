# -*- coding: utf-8 -*-
import os
import time
from typing import Dict


def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def make_output_dir(
    output_dir: str, model_id: str, dataset_info: Dict, encoder_id: str, approach: str
) -> str:
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

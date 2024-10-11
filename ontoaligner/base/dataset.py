# -*- coding: utf-8 -*-
import os
from abc import ABC
from typing import Any, Dict

from ontoaligner.base.ontology import BaseAlignmentsParser
from ontoaligner.utils import io



class OMDataset(ABC):
    track: str = ""
    ontology_name: str = ""

    source_ontology: Any = None
    target_ontology: Any = None

    alignments: Any = BaseAlignmentsParser()

    working_dir: str = ""

    def collect(self, source_ontology_path:str, target_ontology_path:str, reference_matching_path:str) -> Dict:
        data = {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source": self.source_ontology.parse(input_file_path=source_ontology_path),
            "target": self.target_ontology.parse(input_file_path=target_ontology_path),
            "reference": self.alignments.parse(input_file_path=reference_matching_path),
        }
        return data

    def load_from_json(self, root_dir: str) -> Dict:
        json_file_path = os.path.join(
            root_dir, self.track, self.ontology_name, "om.json"
        )
        json_data = io.read_json(input_path=json_file_path)
        return json_data

    def __dir__(self):
        return os.path.join(self.track, self.ontology_name)

    def __str__(self):
        return f"{self.ontology_name}"

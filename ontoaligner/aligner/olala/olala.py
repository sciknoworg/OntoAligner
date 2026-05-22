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
from typing import List,Any

from ...base import BaseOMModel

class OLaLaAligner(BaseOMModel):
    def __init__(self, retriever: Any, llm_aligner: Any, hp_aligner: Any, **kwargs) -> None:
        super().__init__(retriever=retriever,
                         llm_aligner=llm_aligner,
                         hp_aligner=hp_aligner,
                         **kwargs)

    def __str__(self):
        """
        Returns a string representation of the OLaLaAligner model.
        """
        return "OLaLaAligner"

    def load(self, llm_path: str, retriever_path: str) -> None:
        self.kwargs['llm_aligner'].load(path=llm_path)
        self.kwargs['retriever'].load(path=retriever_path)


    def generate(self, input_data: List) -> List:
        """
        Generates ontology alignment results by chaining retrieval, LLM verification,
        and high-precision matching.

        Args:
            input_data (List): A list containing the encoded source and target ontologies.

        Returns:
            List: The combined alignments from LLM and high-precision matching,
                  each annotated with an alignment_type field.
        """
        source_onto, target_onto = input_data[0], input_data[1]

        retriever_candidates = self.kwargs['retriever'].generate(input_data=[source_onto, target_onto])

        llm_alignments = self.kwargs['llm_aligner'].generate(input_data=[source_onto,
                                                                          target_onto,
                                                                          retriever_candidates])

        hp_alignments = self.kwargs['hp_aligner'].generate(input_data=[source_onto, target_onto])

        alignments = [{"alignment_type": "rag", **alignment} for alignment in llm_alignments] + \
                     [{"alignment_type": "hp", **alignment} for alignment in hp_alignments]

        return alignments

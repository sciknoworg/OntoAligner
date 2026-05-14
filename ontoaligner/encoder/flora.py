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
from typing import Any, List

from ..base import BaseEncoder


class FLORAEncoder(BaseEncoder):
    """
    Encoder that prepares pre-loaded FLORA Graph objects for the aligner.

    :class:`~ontoaligner.ontology.flora.FLORAOntology` has already done all the
    heavy lifting (loading the Turtle file and extracting entities / predicates /
    triples).  This encoder's job is to extract the two
    :class:`~ontoaligner.aligner.flora.utils.Graph` objects from the parsed dataset
    dicts and return them as a two-element list, ready for
    :meth:`~ontoaligner.aligner.flora.flora.FLORAAligner.generate`.

    Example:
        >>> # After FLORAOMDataset.collect() has been called:
        >>> encoder = FLORAEncoder()
        >>> kg1_graph, kg2_graph = encoder(source=dataset["source"],
        ...                                target=dataset["target"])
        >>> # kg1_graph and kg2_graph are fully loaded FLORA Graph objects.
    """

    def parse(self, **kwargs) -> List[Any]:
        """
        Extract the two pre-loaded Graph objects from the parser output.

        Args:
            **kwargs: Must contain ``source`` and ``target`` keyword arguments,
                each being the list returned by
                :meth:`~ontoaligner.ontology.flora.FLORAOntology.parse` —
                a single-element list whose first element is a dict with a
                ``"graph"`` key holding the loaded
                :class:`~ontoaligner.aligner.flora.utils.Graph`.

        Returns:
            List[Graph]: ``[kg1_graph, kg2_graph]`` — two fully loaded FLORA
            Graph objects, ready to be passed as ``input_data`` to
            :meth:`~ontoaligner.aligner.flora.flora.FLORAAligner.generate`.
        """
        kg1_graph = kwargs["source"][0]['graph']
        kg2_graph = kwargs["target"][0]['graph']
        return [kg1_graph, kg2_graph]

    def __str__(self) -> str:
        return "FLORAEncoder"

    def get_encoder_info(self) -> str:
        return (
            "FLORA encoder: extracts the pre-loaded FLORA Graph objects from the "
            "structured output of FLORAOntology. The graphs are passed directly to "
            "FLORAAligner.generate(), which runs the fuzzy-logic alignment algorithm."
        )

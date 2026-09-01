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
from io import StringIO
from typing import Any, Dict, List, Optional
import pandas as pd
import bioregistry
from curies import Converter
import sssom as sss
from ontoaligner import __version__


def _get_label_lookup(entities: Optional[List[Dict]]) -> Dict[str, str]:
    """
    Create an IRI-to-label lookup from OntoAligner parser output.

    Parameters:
        entities (Optional[List[Dict]]): Parsed ontology entities containing
            'iri' and optionally 'label'.

    Returns:
        Dict[str, str]: Mapping from entity IRI to label.
    """
    if entities is None:
        return {}

    return {
        entity["iri"]: str(entity["label"])
        for entity in entities
        if "iri" in entity and "label" in entity and entity["label"] is not None
    }

def _get_converter(
    curie_map: Optional[Dict[str, str]] = None,
) -> Converter:
    """
    Get a CURIE converter for SSSOM serialization.

    If no CURIE map is provided, Bioregistry is used as a fallback for
    registered namespaces. Custom or unregistered namespaces must be
    supplied explicitly through ``curie_map``.

    Parameters:
        curie_map (Optional[Dict[str, str]]): Prefix-to-IRI mappings.

    Returns:
        Converter: CURIE converter used for SSSOM serialization.
    """
    if curie_map is not None:
        return Converter.from_prefix_map(curie_map)

    return bioregistry.get_default_converter()

def _get_aligner_metadata(
    matching: Dict,
    aligner: Optional[Any] = None,
    postprocessor: Optional[Any] = None,
    postprocessor_params: Optional[Dict] = None,
) -> Dict:
    """
    Get SSSOM metadata based on the aligner and postprocessor.

    Parameters:
        matching (Dict): OntoAligner matching result.

        aligner (Optional[Any]): Aligner used to generate the matching.

        postprocessor (Optional[Any]): Postprocessor applied to the aligner
            output.

        postprocessor_params (Optional[Dict]): Parameters supplied to the
            postprocessor.

    Returns:
        Dict: SSSOM mapping metadata.
    """
    metadata = {
        "mapping_justification": "semapv:UnspecifiedMatching",
    }

    if aligner is None:
        return metadata

    aligner_name = aligner.__class__.__name__
    postprocessor_name = (
        postprocessor.__name__ if postprocessor is not None else None
    )
    postprocessor_params = postprocessor_params or {}

    score = matching.get("score")
    has_retrieval_threshold = (
        postprocessor_name == "retriever_postprocessor"
        and "threshold" in postprocessor_params
    )

    fuzzy_aligners = {
        "SimpleFuzzySMLightweight": "RapidFuzz fuzz.ratio",
        "WeightedFuzzySMLightweight": "RapidFuzz fuzz.WRatio",
        "TokenSetFuzzySMLightweight": "RapidFuzz fuzz.token_set_ratio",
    }

    graph_aligners = {
        "ConvEAligner",
        "TransDAligner",
        "TransEAligner",
        "TransFAligner",
        "TransHAligner",
        "TransRAligner",
        "DistMultAligner",
        "ComplExAligner",
        "HolEAligner",
        "RotatEAligner",
        "SimplEAligner",
        "CrossEAligner",
        "BoxEAligner",
        "CompGCNAligner",
        "MuREAligner",
        "QuatEAligner",
        "SEAligner",
    }

    if postprocessor_name in {
        "rag_heuristic_postprocessor",
        "rag_hybrid_postprocessor",
    }:
        metadata["mapping_justification"] = "semapv:CompositeMatching"

    elif aligner_name in fuzzy_aligners:
        metadata["mapping_justification"] = (
            "semapv:LexicalSimilarityThresholdMatching"
        )

    elif aligner_name in {"TFIDFRetrieval", "BM25Retrieval"} and has_retrieval_threshold:
        metadata["mapping_justification"] = (
            "semapv:LexicalSimilarityThresholdMatching"
        )

    elif aligner_name in {"SBERTRetrieval", "AdaRetrieval"} and has_retrieval_threshold:
        metadata["mapping_justification"] = (
            "semapv:SemanticSimilarityThresholdMatching"
        )

    elif aligner_name in graph_aligners:
        metadata["mapping_justification"] = "semapv:StructuralMatching"

    elif aligner_name == "PropMatchAligner":
        disable_domain_range = getattr(
            aligner,
            "disable_domain_range",
            getattr(aligner, "kwargs", {}).get("disable_domain_range", False),
        )
        metadata["mapping_justification"] = (
            "semapv:LexicalSimilarityThresholdMatching"
            if disable_domain_range
            else "semapv:CompositeMatching"
        )

    elif aligner_name == "OLaLaHighPrecisionMatcher":
        metadata["mapping_justification"] = "semapv:LexicalMatching"

    elif aligner_name in {"FLORAAligner", "EnsembleLearningAligner"}:
        metadata["mapping_justification"] = "semapv:CompositeMatching"

    if score is not None:
        score = float(score)

        if aligner_name in fuzzy_aligners:
            metadata["similarity_score"] = score
            metadata["similarity_measure"] = fuzzy_aligners[aligner_name]

        elif aligner_name == "TFIDFRetrieval":
            metadata["similarity_score"] = score
            metadata["similarity_measure"] = "TF-IDF cosine similarity"

        elif aligner_name == "SBERTRetrieval" and 0 <= score <= 1:
            metadata["similarity_score"] = score
            metadata["similarity_measure"] = (
                "cosine similarity over SentenceTransformer embeddings"
            )

        elif aligner_name == "AdaRetrieval" and 0 <= score <= 1:
            metadata["similarity_score"] = score
            metadata["similarity_measure"] = (
                "cosine similarity over OpenAI embeddings"
            )

        elif aligner_name in graph_aligners and 0 <= score <= 1:
            metadata["similarity_score"] = score
            metadata["similarity_measure"] = (
                "cosine similarity over graph embeddings"
            )

        elif (
            aligner_name
            in {
                "OLaLaHighPrecisionMatcher",
                "OLaLaLLMAligner",
                "OLaLaAligner",
            }
            and 0 <= score <= 1
        ):
            metadata["confidence"] = score

    if (
        postprocessor_name == "rag_heuristic_postprocessor"
        and matching.get("confidence") is not None
    ):
        confidence = float(matching["confidence"])
        if 0 <= confidence <= 1:
            metadata["confidence"] = confidence

    if (
        postprocessor_name
        in {"rag_heuristic_postprocessor", "rag_hybrid_postprocessor"}
        and score is not None
        and 0 <= score <= 1
    ):
        if "BERTRetriever" in aligner_name:
            metadata["similarity_score"] = score
            metadata["similarity_measure"] = (
                "cosine similarity over SentenceTransformer embeddings"
            )
        elif "AdaRetriever" in aligner_name:
            metadata["similarity_score"] = score
            metadata["similarity_measure"] = (
                "cosine similarity over OpenAI embeddings"
            )

    return metadata


def sssom_alignment_generator(
    matchings: List[Dict],
    source: Optional[List[Dict]] = None,
    target: Optional[List[Dict]] = None,
    *,
    predicate_id: str,
    mapping_set_metadata: Dict,
    curie_map: Optional[Dict[str, str]] = None,
    pipeline: Optional[Any] = None,
    aligner: Optional[Any] = None,
    postprocessor: Optional[Any] = None,
    postprocessor_params: Optional[Dict] = None,
    mapping_justification: Optional[str] = None,
    include_aligner_metadata: bool = True,
 ) -> str:
    """
    Generate ontology alignments in SSSOM TSV format.

    Parameters:
        matchings (List[Dict]): OntoAligner matching results containing
            'source', 'target', and optionally score information.

        source (Optional[List[Dict]]): Parsed source ontology entities.
            Entities containing 'iri' and 'label' are used to populate
            'subject_label'.

        target (Optional[List[Dict]]): Parsed target ontology entities.
            Entities containing 'iri' and 'label' are used to populate
            'object_label'.

        predicate_id (str): SSSOM mapping predicate.

        mapping_set_metadata (Dict): SSSOM MappingSet metadata. It must
            contain 'mapping_set_id' and 'license'.

        curie_map(Optional[Dict[str, str]]): Prefix-to-IRI mappings used to convert
            OntoAligner entity IRIs into SSSOM CURIEs.

        pipeline (Optional[Any]): AlignerPipeline used to generate the
            mappings.

        aligner (Optional[Any]): Aligner used when mappings were generated
            without AlignerPipeline.

        postprocessor (Optional[Any]): Postprocessor used when mappings were
            generated without AlignerPipeline.

        postprocessor_params (Optional[Dict]): Parameters supplied to the
            postprocessor.

        mapping_justification (Optional[str]): Explicit SSSOM mapping
            justification. If not supplied, it is inferred from the aligner
            and postprocessor.

    Returns:
        str: Ontology alignments serialized as SSSOM TSV.
    """
    if "mapping_set_id" not in mapping_set_metadata:
        raise ValueError("'mapping_set_id' is required in mapping_set_metadata.")

    if "license" not in mapping_set_metadata:
        raise ValueError("'license' is required in mapping_set_metadata.")

    if pipeline is not None:
        aligner = pipeline.reranker or pipeline.aligner
        postprocessor = pipeline.postprocessor
        postprocessor_params = pipeline.postprocessor_params

        if source is None and pipeline.om_dataset is not None:
            source = pipeline.om_dataset.get("source")

        if target is None and pipeline.om_dataset is not None:
            target = pipeline.om_dataset.get("target")

    converter = _get_converter(curie_map=curie_map)

    source_labels = _get_label_lookup(source)
    target_labels = _get_label_lookup(target)

    rows = []

    for matching in matchings:
        source_iri = matching["source"]
        target_iri = matching["target"]

        if include_aligner_metadata:
            aligner_metadata = _get_aligner_metadata(
                matching=matching,
                aligner=aligner,
                postprocessor=postprocessor,
                postprocessor_params=postprocessor_params,
            )
            justification = (
                mapping_justification
                or aligner_metadata.get("mapping_justification", "semapv:UnspecifiedMatching")
            )
        else:
            aligner_metadata = {}
            justification = mapping_justification or "semapv:UnspecifiedMatching"

        row = {
            "subject_id": sss.util.safe_compress(source_iri, converter),
            "predicate_id": sss.util.safe_compress(predicate_id, converter),
            "object_id": sss.util.safe_compress(target_iri, converter),
            "mapping_justification": sss.util.safe_compress(
                justification,
                converter,
            ),
        }

        if source_iri in source_labels:
            row["subject_label"] = source_labels[source_iri]

        if target_iri in target_labels:
            row["object_label"] = target_labels[target_iri]

        if include_aligner_metadata and "similarity_score" in aligner_metadata:
            row["similarity_score"] = aligner_metadata["similarity_score"]
            row["similarity_measure"] = aligner_metadata[
                "similarity_measure"
            ]

        if include_aligner_metadata and "confidence" in aligner_metadata:
            row["confidence"] = aligner_metadata["confidence"]

        rows.append(row)

    metadata = dict(mapping_set_metadata)
    metadata.setdefault("mapping_tool", "OntoAligner")
    metadata.setdefault("mapping_tool_version", __version__)

    if pipeline is not None and pipeline.om_dataset is not None:
        dataset_info = pipeline.om_dataset.get("dataset-info", {})
        ontology_name = dataset_info.get("ontology-name")

        if ontology_name is not None:
            metadata.setdefault("mapping_set_title", ontology_name)

    sssom_df = pd.DataFrame(rows)

    sssom_mapping_set = sss.util.MappingSetDataFrame(
        df=sssom_df,
        converter=converter,
        metadata=metadata,
    )

    sssom_output = StringIO()
    sss.writers.write_tsv(sssom_mapping_set, sssom_output)

    return sssom_output.getvalue()

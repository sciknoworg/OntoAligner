import os
import pytest
from ontoaligner.ontology import OLaLaOMDataset
from ontoaligner.aligner.olala import (
    OLaLaSBERTRetrieval,
    OLaLaLLMAligner,
    OLaLaHighPrecisionMatcher,
    OLaLaAligner,
)
from ontoaligner.aligner.olala.postprocessor import olala_postprocessor


def test_dataset_collect(tmp_path):
    # Use the built-in cmt-conference example dataset (OWL/XML)
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'cmt-conference'))
    source = os.path.join(base, "source.xml")
    target = os.path.join(base, "target.xml")
    dataset = OLaLaOMDataset().collect(
        source_ontology_path=source,
        target_ontology_path=target,
    )
    assert "source" in dataset and "target" in dataset
    assert isinstance(dataset["source"], list) and len(dataset["source"]) > 0
    assert isinstance(dataset["target"], list) and len(dataset["target"]) > 0


def test_highprecision_matcher_basic():
    # Two identical items should match exactly
    items = [
        {"iri": "http://example.org/A", "only_label": "A", "hp_texts": ["a"]},
        {"iri": "http://example.org/B", "only_label": "B", "hp_texts": ["b"]},
    ]
    hp = OLaLaHighPrecisionMatcher(confidence=1.0)
    preds = hp.generate(input_data=[items, items])
    # Expect each item to match itself
    sources = {p["source"]: p["target"] for p in preds}
    assert sources.get("http://example.org/A") == "http://example.org/A"
    assert sources.get("http://example.org/B") == "http://example.org/B"


def test_postprocessor_merges():
    # Prepare one rag and one high-precision alignment for the same source
    alignments = [
        {"alignment_type": "rag", "source": "s", "target-cands": ["t"], "score-cands": [0.8]},
        {"alignment_type": "hp", "source": "s", "target": "t", "score": 1.0},
    ]
    # Dummy encoded ontology with host info
    source_items = [{"expected_host": "example.org"}]
    target_items = [{"expected_host": "example.org"}]
    final = olala_postprocessor(
        alignments,
        [source_items, target_items],
        confidence_threshold=0.5,
        strict_bad_hosts=False,
    )
    # Expect the hp match to be preserved with highest score
    assert any(m["score"] == 1.0 and m["source"] == "s" and m["target"] == "t" for m in final)


def test_orchestrator_without_load_raises():
    retriever = OLaLaSBERTRetrieval(device="cpu", top_k=1, both_directions=False, topk_per_resource=False)
    llm = OLaLaLLMAligner(device="cpu", max_new_tokens=1)
    hp = OLaLaHighPrecisionMatcher(confidence=1.0)
    orchestrator = OLaLaAligner(retriever=retriever, llm_aligner=llm, hp_aligner=hp)
    # Without loading models, generate should raise
    with pytest.raises(Exception):
        orchestrator.generate(input_data=[[], []])


def test_retriever_without_load_raises():
    # SBERT retriever must be loaded before generating
    retriever = OLaLaSBERTRetrieval(device="cpu", top_k=1, both_directions=False, topk_per_resource=False)
    with pytest.raises(Exception):
        retriever.generate(input_data=[[], []])

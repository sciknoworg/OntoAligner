import pytest
from ontoaligner.aligner.graph import GraphEmbeddingAligner

@pytest.fixture
def toy_retriever_ontologies():
    source_onto = {
        "entity2iri": {
            "s1": "http://source.org/1",
            "s2": "http://source.org/2",
        },
        "triplets": [
            ("s1", "relatedTo", "s2")
        ],
    }

    target_onto = {
        "entity2iri": {
            "t1": "http://target.org/1",
            "t2": "http://target.org/2",
            "t3": "http://target.org/3",
        },
        "triplets": [
            ("t1", "relatedTo", "t2"),
            ("t2", "relatedTo", "t3"),
        ],
    }

    return source_onto, target_onto


def test_retriever_topk_output(toy_retriever_ontologies):
    source_onto, target_onto = toy_retriever_ontologies

    class DummyAligner(GraphEmbeddingAligner):
        model = "TransE"   # keep it lightweight for test

    # retriever=True → returns top-k candidates
    aligner = DummyAligner(retriever=True, top_k=2, num_epochs=1, embedding_dim=16)

    results = aligner.generate([source_onto, target_onto])

    # Check output format
    assert isinstance(results, list)
    assert all("source" in match for match in results)
    assert all("target-cands" in match for match in results)
    assert all("score-cands" in match for match in results)

    # Check top-k constraint
    for match in results:
        assert len(match["target-cands"]) == 2
        assert len(match["score-cands"]) == 2
        assert all(isinstance(score, float) for score in match["score-cands"])


@pytest.fixture
def toy_kge_ontologies():
    source_onto = {
        "entity2iri": {
            "s1": "http://source.org/1",
            "s2": "http://source.org/2",
        },
        "triplets": [
            ("s1", "relatedTo", "s2")
        ],
    }

    target_onto = {
        "entity2iri": {
            "t1": "http://target.org/1",
            "t2": "http://target.org/2",
        },
        "triplets": [
            ("t1", "relatedTo", "t2")
        ],
    }

    return source_onto, target_onto



def test_kge_aligner_output(toy_kge_ontologies):
    source_onto, target_onto = toy_kge_ontologies

    class DummyAligner(GraphEmbeddingAligner):
        model = "TransE"   # keep it light for testing

    # retriever=False → one-to-one mapping
    aligner = DummyAligner(retriever=False, num_epochs=1, embedding_dim=16)

    results = aligner.generate([source_onto, target_onto])

    # Check output format
    assert isinstance(results, list)
    assert all(isinstance(match, dict) for match in results)

    # Check required keys
    for match in results:
        assert "source" in match
        assert "target" in match
        assert "score" in match

        # Check value types
        assert isinstance(match["source"], str)
        assert isinstance(match["target"], str)
        assert isinstance(match["score"], float)

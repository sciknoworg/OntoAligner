import unittest

from ontoaligner.aligner.retrieval.reranking import Reranking, CohereReranking, CrossEncoderReranking
from ontoaligner.postprocess import retriever_postprocessor


class DummyReranking(Reranking):
    def rerank_candidates(self, query, documents):
        return [0.1, 0.9, 0.5][:len(documents)]


class FakeCohereResult:
    def __init__(self, index, relevance_score):
        self.index = index
        self.relevance_score = relevance_score


class FakeCohereResponse:
    def __init__(self, results):
        self.results = results


class FakeCohereClient:
    def rerank(self, model, query, documents, top_n):
        return FakeCohereResponse(
            [
                FakeCohereResult(index=1, relevance_score=0.8),
                FakeCohereResult(index=0, relevance_score=0.3),
            ]
        )


class FakeCrossEncoderModel:
    def predict(self, pairs, batch_size=16):
        return [0.2, 1.4, -0.5][:len(pairs)]


class TestReranking(unittest.TestCase):
    def setUp(self):
        self.source_ontology = [
            {
                "iri": "http://example.org/source1",
                "text": "Computer Science",
                "label": "Computer Science",
            }
        ]

        self.target_ontology = [
            {
                "iri": "http://example.org/target1",
                "text": "Computer Sciences",
                "label": "Computer Sciences",
            },
            {
                "iri": "http://example.org/target2",
                "text": "Information Technology",
                "label": "Information Technology",
            },
            {
                "iri": "http://example.org/target3",
                "text": "Software Engineering",
                "label": "Software Engineering",
            },
        ]

        self.retrieval_outputs = [
            {
                "source": "http://example.org/source1",
                "target-cands": [
                    "http://example.org/target1",
                    "http://example.org/target2",
                    "http://example.org/target3",
                ],
                "score-cands": [0.7, 0.6, 0.4],
            }
        ]

    def test_reranking_initialization(self):
        reranker = DummyReranking(device="cpu", top_k=2, normalize_score="none")

        self.assertEqual(reranker.kwargs["device"], "cpu")
        self.assertEqual(reranker.kwargs["top_k"], 2)
        self.assertEqual(reranker.kwargs["normalize_score"], "none")
        self.assertEqual(str(reranker), "Reranking")

    def test_reranking_generate(self):
        reranker = DummyReranking(top_k=2, normalize_score="none")

        predictions = reranker.generate(
            [
                self.source_ontology,
                self.target_ontology,
                self.retrieval_outputs,
            ]
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["source"], "http://example.org/source1")
        self.assertEqual(
            predictions[0]["target-cands"],
            [
                "http://example.org/target2",
                "http://example.org/target3",
            ],
        )
        self.assertEqual(predictions[0]["score-cands"], [0.9, 0.5])
        self.assertEqual(predictions[0]["retriever-score-cands"], [0.6, 0.4])

    def test_reranking_with_retriever_postprocessor(self):
        reranker = DummyReranking(top_k=2, normalize_score="none")

        predictions = reranker.generate(
            [
                self.source_ontology,
                self.target_ontology,
                self.retrieval_outputs,
            ]
        )

        matchings = retriever_postprocessor(predictions, threshold=0.6)

        self.assertEqual(len(matchings), 1)
        self.assertEqual(matchings[0]["source"], "http://example.org/source1")
        self.assertEqual(matchings[0]["target"], "http://example.org/target2")
        self.assertEqual(matchings[0]["score"], 0.9)

    def test_score_normalization(self):
        reranker = DummyReranking(normalize_score="sigmoid")
        scores = reranker.normalize([0.0])

        self.assertAlmostEqual(scores[0], 0.5)

    def test_cohere_reranking(self):
        reranker = CohereReranking(cohere_key="test-key", top_k=2)
        reranker.path = "rerank-v3.5"
        reranker.model = FakeCohereClient()

        scores = reranker.rerank_candidates(
            query="Computer Science",
            documents=["Computer Sciences", "Information Technology"],
        )

        self.assertEqual(str(reranker), "Reranking+CohereReranking")
        self.assertEqual(reranker.kwargs["normalize_score"], "none")
        self.assertEqual(scores, [0.3, 0.8])

    def test_cross_encoder_reranking(self):
        reranker = CrossEncoderReranking(device="cpu", top_k=2)
        reranker.model = FakeCrossEncoderModel()

        scores = reranker.rerank_candidates(
            query="Computer Science",
            documents=[
                "Computer Sciences",
                "Information Technology",
                "Software Engineering",
            ],
        )

        self.assertEqual(str(reranker), "Reranking+CrossEncoderReranking")
        self.assertEqual(reranker.kwargs["normalize_score"], "sigmoid")
        self.assertEqual(scores, [0.2, 1.4, -0.5])


if __name__ == "__main__":
    unittest.main()
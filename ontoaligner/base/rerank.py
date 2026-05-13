from abc import ABC, abstractmethod
from typing import List, Dict, Any
from tqdm import tqdm

class Rerank(ABC):
    """
    Abstract base class for reranking ontology alignment candidates.

    Subclasses must implement `rerank_candidates` which, given a source text
    and a list of candidate target texts, returns them reranked with scores.
    """

    def __init__(self, top_n: int = 1):
        """
        Args:
            top_n: Number of top candidates to return per source after reranking.
        """
        self.top_n = top_n

    @abstractmethod
    def rerank_candidates(
        self, query: str, documents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate documents against a query.

        Args:
            query: The source concept text.
            documents: List of candidate target concept texts.

        Returns:
            List of dicts with keys:
                - "index": original index in the documents list
                - "relevance_score": reranking score
        """
        pass

    def rerank_retrieval_outputs(
        self,
        retrieval_outputs: List[Dict],
        source_iri2text: Dict[str, str],
        target_iri2text: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Rerank all retrieval outputs and return flat source-target-score predictions.

        Args:
            retrieval_outputs: BM25 outputs with "source", "target-cands", "score-cands".
            source_iri2text: Mapping from source IRI to text label.
            target_iri2text: Mapping from target IRI to text label.

        Returns:
            List of {"source": iri, "target": iri, "score": float}
        """
        predictions = []
        for entry in tqdm(retrieval_outputs, desc="Reranking"):
            source_iri = entry["source"]
            candidate_iris = entry["target-cands"]

            query_text = source_iri2text.get(source_iri, "")
            doc_texts = [target_iri2text.get(iri, "") for iri in candidate_iris]

            if not query_text or not doc_texts:
                continue

            ranked = self.rerank_candidates(query=query_text, documents=doc_texts)

            for item in ranked:
                idx = item["index"]
                predictions.append({
                    "source": source_iri,
                    "target": candidate_iris[idx],
                    "score": item["relevance_score"],
                })

        return predictions

class CohereRerank(Rerank):
    """
    Reranker using the Cohere Rerank API.

    See: https://docs.cohere.com/reference/rerank
    """

    def __init__(self, api_key: str, model: str = "rerank-v3.5", top_n: int = 1):
        """
        Args:
            api_key: Cohere API key.
            model: Cohere rerank model name.
            top_n: Number of top candidates to return per query.
        """
        super().__init__(top_n=top_n)
        import cohere
        self.client = cohere.ClientV2(api_key)
        self.model = model

    def rerank_candidates(
        self, query: str, documents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Calls the Cohere Rerank API to rerank candidate documents.

        Args:
            query: The source concept text.
            documents: List of candidate target concept texts.

        Returns:
            List of dicts with "index" and "relevance_score",
            limited to self.top_n results.
        """
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=self.top_n,
        )
        return [
            {"index": r.index, "relevance_score": r.relevance_score}
            for r in response.results
        ]
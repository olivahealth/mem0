from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from mem0.types import VectorStoreResult

class Reranker:
    @staticmethod
    def rerank_results(
        query: str,
        results: List[VectorStoreResult],
        field: str = "data"
    ) -> List[VectorStoreResult]:
        """
        Rerank results using BM25 algorithm.
        
        Args:
            query: The search query
            results: List of vector search results
            field: The field in payload to use for text comparison
        
        Returns:
            Reranked list of results
        """
        if not results:
            return results

        # Tokenize query and documents
        query_tokens = query.lower().split()
        documents = [
            str(result.payload.get(field, "")).lower().split()
            for result in results
        ]
        
        # Create BM25 instance
        bm25 = BM25Okapi(documents)
        
        # Get BM25 scores
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Combine vector similarity scores with BM25 scores
        # Using weighted average (0.7 for vector similarity, 0.3 for BM25)
        combined_results = []
        for idx, (result, bm25_score) in enumerate(zip(results, bm25_scores)):
            vector_score = result.score
            # Normalize BM25 score to 0-1 range
            normalized_bm25_score = bm25_score / max(bm25_scores) if max(bm25_scores) > 0 else 0
            # Combine scores
            combined_score = (0.7 * vector_score) + (0.3 * normalized_bm25_score)
            combined_results.append((result, combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return reranked results with updated scores
        return [
            VectorStoreResult(
                id=result.id,
                payload=result.payload,
                score=combined_score
            )
            for result, combined_score in combined_results
        ]
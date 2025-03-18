from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, name, vector_size, distance):
        """Create a new collection."""
        pass

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""
        pass

    @abstractmethod
    def search(self, query, limit=5, filters=None):
        """Search for similar vectors."""
        pass

    def search_with_rerank(
        self,
        query: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Search with optional reranking.
        
        Args:
            query: Original text query for reranking
            query_vector: Embedded query vector
            limit: Number of results to return
            filters: Optional filters to apply
            rerank: Whether to apply reranking
            
        Returns:
            List of search results
        """
        # Get initial vector search results
        results = self.search(
            query=query_vector,
            limit=limit * 2 if rerank else limit,  # Get more results if reranking
            filters=filters
        )
        
        # Apply reranking if enabled
        if rerank and len(results) > 0:
            from rank_bm25 import BM25Okapi
            import numpy as np
            
            # Prepare documents for BM25
            documents = [
                str(result.payload.get('data', '')).lower().split()
                for result in results
            ]
            query_tokens = query.lower().split()
            
            # Calculate BM25 scores
            bm25 = BM25Okapi(documents)
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Handle numpy array properly, avoid division by zero
            max_bm25_score = float(np.max(bm25_scores)) if len(bm25_scores) > 0 else 1.0
            if max_bm25_score == 0:
                max_bm25_score = 1.0
            
            # Combine scores
            for idx, (result, bm25_score) in enumerate(zip(results, bm25_scores)):
                vector_score = result.score
                normalized_bm25_score = float(bm25_score) / max_bm25_score
                combined_score = (0.7 * vector_score) + (0.3 * normalized_bm25_score)
                results[idx].score = combined_score
            
            # Sort by combined score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:limit]
            
        return results

    @abstractmethod
    def delete(self, vector_id):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def update(self, vector_id, vector=None, payload=None):
        """Update a vector and its payload."""
        pass

    @abstractmethod
    def get(self, vector_id):
        """Retrieve a vector by ID."""
        pass

    @abstractmethod
    def list_cols(self):
        """List all collections."""
        pass

    @abstractmethod
    def delete_col(self):
        """Delete a collection."""
        pass

    @abstractmethod
    def col_info(self):
        """Get information about a collection."""
        pass

    @abstractmethod
    def list(self, filters=None, limit=None):
        """List all memories."""
        pass

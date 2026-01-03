"""RAG (Retrieval-Augmented Generation) module using Qdrant Cloud.

This module provides vector search capabilities for product knowledge
and documentation using Qdrant Cloud as the vector database.
"""

import hashlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config import config


class QdrantRAG:
    """RAG implementation using Qdrant Cloud for vector storage."""
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize Qdrant Cloud client.
        
        Args:
            url: Qdrant Cloud URL (e.g., https://xxx.cloud.qdrant.io)
            api_key: Qdrant Cloud API key
            collection_name: Name of the collection to use
            embedding_model: Sentence transformer model for embeddings
        """
        self.url = url or config.qdrant_url
        self.api_key = api_key or config.qdrant_api_key
        self.collection_name = collection_name or config.qdrant_collection
        
        # Initialize Qdrant client for cloud
        if self.url and self.api_key:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
        else:
            # Fallback to in-memory for testing
            self.client = QdrantClient(":memory:")
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self._embedder = None
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
    
    @property
    def embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            self.vector_size = self._embedder.get_sentence_embedding_dimension()
        return self._embedder
    
    def _generate_id(self, text: str) -> str:
        """Generate a deterministic ID from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return True
        return False
    
    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Add documents to the collection.
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata' keys
            batch_size: Number of documents to process at once
            
        Returns:
            Number of documents added
        """
        self.ensure_collection()
        
        points = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Generate embedding
            embedding = self.embedder.encode(text).tolist()
            
            # Create point
            point_id = self._generate_id(text)
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": text,
                    **metadata,
                },
            ))
        
        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        return len(points)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        filter_conditions: dict[str, Any] = None,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filter_conditions: Optional filter conditions (field: value)
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Build filter if provided
        search_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            search_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter,
            score_threshold=score_threshold,
        )
        
        # Format results
        return [
            {
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            }
            for hit in results
        ]
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            return {"error": str(e)}


class ProductKnowledgeBase:
    """Knowledge base for product information using RAG."""
    
    def __init__(self, qdrant_rag: QdrantRAG = None):
        self.rag = qdrant_rag or QdrantRAG()
    
    def index_products(self, products: list[dict]) -> int:
        """Index product information for retrieval.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Number of documents indexed
        """
        documents = []
        for product in products:
            # Create searchable text from product info
            text = f"""
Product: {product.get('name', '')}
Category: {product.get('category', '')}
Description: {product.get('description', '')}
Price: ${product.get('price', 0)}
Specifications: {product.get('specs', {})}
"""
            documents.append({
                "text": text.strip(),
                "metadata": {
                    "product_id": product.get("product_id", ""),
                    "name": product.get("name", ""),
                    "category": product.get("category", ""),
                    "price": product.get("price", 0),
                    "in_stock": product.get("in_stock", False),
                },
            })
        
        return self.rag.add_documents(documents)
    
    def search_products(
        self,
        query: str,
        category: str = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search for products matching a query.
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of matching products with relevance scores
        """
        filters = {}
        if category:
            filters["category"] = category
        
        return self.rag.search(
            query=query,
            limit=limit,
            filter_conditions=filters if filters else None,
        )
    
    def index_faqs(self, faqs: list[dict]) -> int:
        """Index FAQ content for retrieval.
        
        Args:
            faqs: List of FAQ dictionaries with 'topic' and 'content'
            
        Returns:
            Number of documents indexed
        """
        documents = []
        for faq in faqs:
            text = f"Topic: {faq.get('topic', '')}\n\n{faq.get('content', '')}"
            documents.append({
                "text": text,
                "metadata": {
                    "type": "faq",
                    "topic": faq.get("topic", ""),
                },
            })
        
        return self.rag.add_documents(documents)
    
    def search_faqs(self, query: str, limit: int = 3) -> list[dict]:
        """Search FAQs for relevant answers."""
        return self.rag.search(
            query=query,
            limit=limit,
            filter_conditions={"type": "faq"},
        )


# Singleton instance
_rag_instance: QdrantRAG | None = None


def get_rag() -> QdrantRAG:
    """Get or create the RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = QdrantRAG()
    return _rag_instance

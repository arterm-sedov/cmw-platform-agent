"""
Similarity Manager Module

This module handles all vector similarity operations without requiring Supabase
or external services. It provides a clean, standalone interface for text similarity
and vector-based operations.

Features:
- Local embeddings using sentence-transformers (optional)
- Cosine similarity calculations between texts
- Vector-based answer matching
- Text normalization and preprocessing
- Fallback to simple text similarity when embeddings unavailable

Usage:
    from similarity_manager import SimilarityManager
    manager = SimilarityManager()
    similarity = manager.calculate_similarity(text1, text2)
    is_match, score = manager.vector_answers_match(answer, reference)
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import hashlib

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

class SimilarityManager:
    """
    Manages all similarity operations including embeddings and vector-based
    text matching without external dependencies.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", enabled: bool = True):
        """
        Initialize the similarity manager.

        Args:
            model_name: Name of the sentence transformer model to use
            enabled: Whether to enable vector similarity (loads heavy models if True)
        """
        self.model_name = model_name
        self.model = None
        self.enabled = enabled
        if self.enabled:
            self._load_model()
        else:
            print("ℹ️ Vector similarity disabled - skipping heavy model loading")

    def _load_model(self):
        """Load the sentence transformer model if available."""
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
                print(f"✅ Loaded similarity model: {self.model_name}")
            except Exception as e:
                print(f"⚠️ Failed to load similarity model: {e}")
                self.model = None
        else:
            print("⚠️ Sentence transformers not available - using fallback methods")

    def is_enabled(self) -> bool:
        """Check if the similarity manager is enabled and working."""
        return self.enabled and self.model is not None

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using sentence-transformers.

        Args:
            text: Input text to embed

        Returns:
            Numpy array embedding or None if failed
        """
        if not self.is_enabled() or not text:
            return None

        try:
            # Clean and normalize text
            text = self._normalize_text(text)
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            return None

    def embed_query(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a text query.

        This is the LangChain-compatible interface that returns List[float].

        Args:
            text (str): Text to embed

        Returns:
            Embedding vector as List[float] or None if disabled
        """
        embedding = self.generate_embedding(text)
        if embedding is not None:
            return embedding.tolist()
        return None

    def calculate_cosine_similarity(self, embedding1, embedding2) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector (list, numpy array, or similar)
            embedding2: Second embedding vector (list, numpy array, or similar)

        Returns:
            float: Cosine similarity score between 0.0 and 1.0
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception as e:
            print(f"❌ Failed to calculate cosine similarity: {e}")
            return 0.0

    def get_embeddings(self):
        """
        Get the embeddings instance for compatibility with LangChain interfaces.

        Returns:
            Embeddings instance or None if not available
        """
        return self.model if self.is_enabled() else None

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0

        # Fallback to simple text similarity if model not available
        if not self.is_enabled():
            return self._simple_text_similarity(text1, text2)

        try:
            emb1 = self.generate_embedding(text1)
            emb2 = self.generate_embedding(text2)

            if emb1 is None or emb2 is None:
                return self._simple_text_similarity(text1, text2)

            return self._cosine_similarity(emb1, emb2)
        except Exception as e:
            print(f"❌ Failed to calculate similarity: {e}")
            return self._simple_text_similarity(text1, text2)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """
        Simple text similarity fallback using Jaccard similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Normalize texts
            text1 = self._normalize_text(text1).lower()
            text2 = self._normalize_text(text2).lower()

            # Simple word overlap similarity
            words1 = set(text1.split())
            words2 = set(text2.split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union)
        except Exception:
            return 0.0

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better similarity calculation."""
        if not isinstance(text, str):
            text = str(text)

        # Remove extra whitespace and normalize
        text = " ".join(text.split())

        return text


    def vector_answers_match(self, answer: str, reference: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """
        Check if two answers match using vector similarity.

        Args:
            answer: Generated answer
            reference: Reference answer
            threshold: Similarity threshold

        Returns:
            Tuple of (is_match, similarity_score)
        """
        if not answer or not reference:
            return False, 0.0

        try:
            similarity = self.calculate_similarity(answer, reference)
            is_match = similarity >= threshold
            return is_match, similarity
        except Exception as e:
            print(f"❌ Failed to compare answers: {e}")
            return False, 0.0


    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the similarity manager."""
        return {
            "enabled": self.is_enabled(),
            "model_name": self.model_name,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }


# Global instance for easy access - will be initialized lazily
similarity_manager = None

def get_similarity_manager(enabled: bool = True):
    """Get or create the global similarity manager instance."""
    global similarity_manager
    if similarity_manager is None:
        similarity_manager = SimilarityManager(enabled=enabled)
    return similarity_manager

# Convenience functions for similarity operations
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    if similarity_manager is None:
        get_similarity_manager()  # Initialize with default enabled=True
    return similarity_manager.calculate_similarity(text1, text2)

def vector_answers_match(answer: str, reference: str, threshold: float = 0.8) -> Tuple[bool, float]:
    """Check if answers match using vector similarity."""
    if similarity_manager is None:
        get_similarity_manager()  # Initialize with default enabled=True
    return similarity_manager.vector_answers_match(answer, reference, threshold)

def calculate_cosine_similarity(embedding1, embedding2) -> float:
    """Calculate cosine similarity between two embeddings."""
    if similarity_manager is None:
        get_similarity_manager()  # Initialize with default enabled=True
    return similarity_manager.calculate_cosine_similarity(embedding1, embedding2)

def embed_query(text: str) -> Optional[List[float]]:
    """Generate embedding for a text query."""
    if similarity_manager is None:
        get_similarity_manager()  # Initialize with default enabled=True
    return similarity_manager.embed_query(text)

def get_embeddings():
    """Get the embeddings instance."""
    if similarity_manager is None:
        get_similarity_manager()  # Initialize with default enabled=True
    return similarity_manager.get_embeddings()

def get_similarity_status() -> Dict[str, Any]:
    """Get similarity manager status."""
    return similarity_manager.get_status()

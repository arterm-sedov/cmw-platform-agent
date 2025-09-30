"""
Vector Similarity Module
========================

This module handles all vector similarity calculations and comparisons.
It provides utilities for embedding generation, similarity scoring, and answer matching.

Key Features:
- Cosine similarity calculations
- Embedding generation and comparison
- Answer similarity matching
- Tool call deduplication
- Vector store integration

Usage:
    from vector_similarity import VectorSimilarityManager
    
    vsm = VectorSimilarityManager()
    similarity = vsm.calculate_similarity(text1, text2)
    is_match = vsm.answers_match(answer, reference)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class SimilarityResult:
    """Represents a similarity calculation result"""
    is_match: bool
    similarity_score: float
    threshold: float
    method: str


class VectorSimilarityManager:
    """Manages vector similarity calculations and comparisons"""
    
    def __init__(self, enabled: bool = True, similarity_threshold: float = 0.95, 
                 tool_calls_threshold: float = 0.90):
        self.enabled = enabled
        self.similarity_threshold = similarity_threshold
        self.tool_calls_threshold = tool_calls_threshold
        self.embedding_model = None
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """Initialize embedding model if vector similarity is enabled"""
        if not self.enabled:
            return
        
        try:
            # Try to import and initialize embedding model
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("[VectorSimilarity] Warning: sentence-transformers not available, using fallback")
            self.embedding_model = None
        except Exception as e:
            print(f"[VectorSimilarity] Warning: Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        If vector similarity is disabled, returns 1.0 to indicate perfect match.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (0.0 to 1.0)
        """
        if not self.enabled or embedding1 is None or embedding2 is None:
            return 1.0
        
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        except Exception as e:
            print(f"[VectorSimilarity] Error calculating cosine similarity: {e}")
            return 0.0
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector or None if failed
        """
        if not self.enabled or not text or not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"[VectorSimilarity] Error generating embedding: {e}")
            return None
    
    def answers_match(self, answer: str, reference: str) -> Tuple[bool, float]:
        """
        Check if two answers match using vector similarity.
        If vector similarity is disabled, returns (True, 1.0) to always consider answers as matching.
        
        Args:
            answer: Answer to compare
            reference: Reference answer
            
        Returns:
            Tuple[bool, float]: (is_match, similarity_score)
        """
        if not self.enabled:
            return True, 1.0
        
        if not answer or not reference:
            return False, 0.0
        
        try:
            # Generate embeddings
            answer_embedding = self.generate_embedding(answer)
            reference_embedding = self.generate_embedding(reference)
            
            if answer_embedding is None or reference_embedding is None:
                # Fallback to text similarity
                return self._text_similarity_match(answer, reference)
            
            # Calculate similarity
            similarity = self.calculate_cosine_similarity(answer_embedding, reference_embedding)
            is_match = similarity >= self.similarity_threshold
            
            return is_match, similarity
        
        except Exception as e:
            print(f"[VectorSimilarity] Error in answers_match: {e}")
            # Fallback to text similarity
            return self._text_similarity_match(answer, reference)
    
    def tool_calls_match(self, tool_call1: Dict[str, Any], tool_call2: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if two tool calls are similar for deduplication.
        
        Args:
            tool_call1: First tool call
            tool_call2: Second tool call
            
        Returns:
            Tuple[bool, float]: (is_match, similarity_score)
        """
        if not self.enabled:
            return False, 0.0
        
        try:
            # Extract tool call information
            name1 = tool_call1.get('name', '')
            name2 = tool_call2.get('name', '')
            
            # Different tools are never similar
            if name1 != name2:
                return False, 0.0
            
            # Compare arguments
            args1 = tool_call1.get('args', {})
            args2 = tool_call2.get('args', {})
            
            # Convert args to strings for comparison
            args_str1 = str(sorted(args1.items()))
            args_str2 = str(sorted(args2.items()))
            
            # Generate embeddings for arguments
            args_embedding1 = self.generate_embedding(args_str1)
            args_embedding2 = self.generate_embedding(args_str2)
            
            if args_embedding1 is None or args_embedding2 is None:
                # Fallback to exact string match
                return args_str1 == args_str2, 1.0 if args_str1 == args_str2 else 0.0
            
            # Calculate similarity
            similarity = self.calculate_cosine_similarity(args_embedding1, args_embedding2)
            is_match = similarity >= self.tool_calls_threshold
            
            return is_match, similarity
        
        except Exception as e:
            print(f"[VectorSimilarity] Error in tool_calls_match: {e}")
            return False, 0.0
    
    def _text_similarity_match(self, text1: str, text2: str) -> Tuple[bool, float]:
        """
        Fallback text similarity matching using simple string operations.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Tuple[bool, float]: (is_match, similarity_score)
        """
        if not text1 or not text2:
            return False, 0.0
        
        # Normalize texts
        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)
        
        # Calculate Jaccard similarity
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if not words1 or not words2:
            return False, 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Use a lower threshold for text similarity
        text_threshold = self.similarity_threshold * 0.8
        is_match = jaccard_similarity >= text_threshold
        
        return is_match, jaccard_similarity
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def find_similar_answers(self, answer: str, answer_list: List[str], 
                           threshold: float = None) -> List[Tuple[str, float]]:
        """
        Find similar answers in a list.
        
        Args:
            answer: Answer to compare against
            answer_list: List of answers to search
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            List[Tuple[str, float]]: List of (answer, similarity) tuples
        """
        if not self.enabled or not answer or not answer_list:
            return []
        
        threshold = threshold or self.similarity_threshold
        similar_answers = []
        
        for candidate in answer_list:
            is_match, similarity = self.answers_match(answer, candidate)
            if similarity >= threshold:
                similar_answers.append((candidate, similarity))
        
        # Sort by similarity (highest first)
        similar_answers.sort(key=lambda x: x[1], reverse=True)
        
        return similar_answers
    
    def deduplicate_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate tool calls based on similarity.
        
        Args:
            tool_calls: List of tool calls
            
        Returns:
            List[Dict[str, Any]]: Deduplicated tool calls
        """
        if not self.enabled or len(tool_calls) <= 1:
            return tool_calls
        
        deduplicated = []
        
        for tool_call in tool_calls:
            is_duplicate = False
            
            for existing in deduplicated:
                is_match, similarity = self.tool_calls_match(tool_call, existing)
                if is_match:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(tool_call)
        
        return deduplicated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector similarity manager statistics"""
        return {
            'enabled': self.enabled,
            'similarity_threshold': self.similarity_threshold,
            'tool_calls_threshold': self.tool_calls_threshold,
            'embedding_model_available': self.embedding_model is not None,
            'embedding_model_name': 'all-MiniLM-L6-v2' if self.embedding_model else None
        }


# Global vector similarity manager instance
_vector_similarity_manager = None

def get_vector_similarity_manager(enabled: bool = True) -> VectorSimilarityManager:
    """Get the global vector similarity manager instance"""
    global _vector_similarity_manager
    if _vector_similarity_manager is None or _vector_similarity_manager.enabled != enabled:
        _vector_similarity_manager = VectorSimilarityManager(enabled=enabled)
    return _vector_similarity_manager

def reset_vector_similarity_manager():
    """Reset the global vector similarity manager instance"""
    global _vector_similarity_manager
    _vector_similarity_manager = None

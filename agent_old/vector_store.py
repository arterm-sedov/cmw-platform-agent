"""
Vector Store Management Module

This module handles all Supabase vector store operations including:
- Connection management
- Embedding generation
- Similarity search
- Reference answer retrieval
- Tool call deduplication

The module is designed to be disabled when Supabase is not available.
"""

import os
import json
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Import similarity manager for similarity operations
from similarity_manager import similarity_manager

# Global flag to enable Supabase functionality if environment variables are set
import os
SUPABASE_ENABLED = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))

# Try to import Supabase dependencies
try:
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from supabase.client import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: Supabase dependencies not available. Install with: pip install supabase langchain-community langchain-huggingface")

class VectorStoreManager:
    """
    Manages Supabase vector store operations with graceful fallbacks.
    
    This class provides a unified interface for vector store operations
    and can be disabled when Supabase is not available or configured.
    """
    
    def __init__(self, enabled: bool = None, enable_vector_similarity: bool = False):
        """
        Initialize the vector store manager.
        
        Args:
            enabled (bool, optional): Override the global SUPABASE_ENABLED setting
            enable_vector_similarity (bool): Whether to enable vector similarity (loads heavy models if True)
        """
        self.enabled = enabled if enabled is not None else SUPABASE_ENABLED
        self.enable_vector_similarity = enable_vector_similarity
        self.embeddings = None
        self.supabase_client = None
        self.vector_store = None
        self.retriever_tool = None
        
        if self.enabled and SUPABASE_AVAILABLE and self.enable_vector_similarity:
            self._initialize_supabase()
        else:
            if not self.enable_vector_similarity:
                print("ℹ️ Vector similarity disabled - skipping heavy model loading")
            else:
                print("ℹ️ Vector store functionality is disabled")
    
    def _initialize_supabase(self):
        """Initialize Supabase client and vector store."""
        try:
            # Check for required environment variables
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                print("⚠️ Supabase environment variables not found. Disabling vector store.")
                self.enabled = False
                return
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            
            # Initialize Supabase client
            self.supabase_client = create_client(supabase_url, supabase_key)
            
            # Initialize vector store
            self.vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name="agent_course_reference",
                quick_name="match_agent_course_reference_langchain",
            )
            
            print("✅ Supabase vector store initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Supabase vector store: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if vector store functionality is enabled."""
        return self.enabled and self.vector_store is not None
    
    
    def get_vector_store(self):
        """Get the vector store instance."""
        return self.vector_store if self.is_enabled() else None
    
    def get_retriever_tool(self):
        """Get the retriever tool instance."""
        if not self.is_enabled():
            return None
            
        try:
            from langchain.tools.retriever import create_retriever_tool
            return create_retriever_tool(
                retriever=self.vector_store.as_retriever(),
                name="Question Search",
                description="A tool to retrieve similar questions from a vector store.",
            )
        except Exception as e:
            print(f"❌ Failed to create retriever tool: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 1) -> List[Any]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List of search results or empty list if disabled
        """
        if not self.is_enabled():
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
            return []
    
    def get_reference_answer(self, question: str) -> Optional[str]:
        """
        Retrieve reference answer for a question.
        
        Args:
            question (str): The question text
            
        Returns:
            Reference answer if found, None otherwise
        """
        if not self.is_enabled():
            return None
        
        try:
            similar = self.similarity_search(question)
            if similar:
                content = similar[0].page_content
                # Try to extract the answer from the content
                if "Final answer :" in content:
                    return content.split("Final answer :", 1)[-1].strip().split("\n")[0]
                return content
            return None
        except Exception as e:
            print(f"❌ Failed to get reference answer: {e}")
            return None
    
    
    def vector_answers_match(self, answer: str, reference: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """
        Check if two answers match using vector similarity.
        
        Args:
            answer (str): First answer
            reference (str): Reference answer
            threshold (float): Similarity threshold
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        if not self.is_enabled():
            return False, 0.0
        
        try:
            # Handle None or empty answers gracefully
            if not answer or not reference:
                return False, 0.0
            
            # Normalize answers
            norm_answer = self._clean_answer_text(answer)
            norm_reference = self._clean_answer_text(reference)
            
            # Check for exact match
            if norm_answer == norm_reference:
                return True, 1.0
            
            # Generate embeddings using similarity manager
            answer_embedding = similarity_manager.embed_query(norm_answer)
            reference_embedding = similarity_manager.embed_query(norm_reference)

            if not answer_embedding or not reference_embedding:
                return False, 0.0

            # Calculate similarity using similarity manager
            similarity = similarity_manager.calculate_cosine_similarity(answer_embedding, reference_embedding)
            return similarity >= threshold, similarity
            
        except Exception as e:
            print(f"❌ Failed to compare answers: {e}")
            return False, 0.0
    
    def _clean_answer_text(self, text: str) -> str:
        """
        Clean and normalize answer text for comparison.
        
        Args:
            text (str): Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove common prefixes and normalize
        text = text.strip().lower()
        
        # Remove "FINAL ANSWER:" prefix if present
        if text.startswith("final answer:"):
            text = text[13:].strip()
        
        # Remove extra whitespace and normalize
        text = " ".join(text.split())
        
        return text

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the vector store manager.
        
        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.enabled,
            "supabase_available": SUPABASE_AVAILABLE,
            "embeddings_initialized": self.embeddings is not None,
            "vector_store_initialized": self.vector_store is not None,
            "supabase_url_configured": bool(os.environ.get("SUPABASE_URL")),
            "supabase_key_configured": bool(os.environ.get("SUPABASE_KEY")),
        }

# Global instance for easy access - will be initialized lazily
vector_store_manager = None

def get_vector_store_manager(enabled: bool = None, enable_vector_similarity: bool = True):
    """Get or create the global vector store manager instance."""
    global vector_store_manager
    if vector_store_manager is None:
        vector_store_manager = VectorStoreManager(enabled=enabled, enable_vector_similarity=enable_vector_similarity)
    return vector_store_manager

# Convenience functions for vector store operations
def get_vector_store():
    """Get vector store instance."""
    if vector_store_manager is None:
        get_vector_store_manager()  # Initialize with defaults
    return vector_store_manager.get_vector_store()

def get_retriever_tool():
    """Get retriever tool instance."""
    if vector_store_manager is None:
        get_vector_store_manager()  # Initialize with defaults
    return vector_store_manager.get_retriever_tool()

def similarity_search(query: str, k: int = 1):
    """Perform similarity search."""
    if vector_store_manager is None:
        get_vector_store_manager()  # Initialize with defaults
    return vector_store_manager.similarity_search(query, k)

def get_reference_answer(question: str):
    """Get reference answer for a question."""
    if vector_store_manager is None:
        get_vector_store_manager()  # Initialize with defaults
    return vector_store_manager.get_reference_answer(question)

def get_status():
    """Get vector store status."""
    if vector_store_manager is None:
        get_vector_store_manager()  # Initialize with defaults
    return vector_store_manager.get_status()

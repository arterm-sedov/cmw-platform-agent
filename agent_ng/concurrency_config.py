"""
Concurrency Configuration Module
===============================

A lean, Pydantic-based configuration system for managing Gradio concurrency settings.
This module provides type-safe, validated configuration for queue management and
concurrent processing in a Gradio-native way.

Key Features:
- Pydantic models for type safety and validation
- Gradio-native concurrency patterns
- Modular configuration management
- Environment variable support
- Default values with sensible overrides
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, validator
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class QueueConfig(BaseModel):
    """Configuration for Gradio queue management"""
    
    default_concurrency_limit: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Default concurrency limit for all event listeners"
    )
    
    max_threads: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of threads for Gradio processing"
    )
    
    enable_queue: bool = Field(
        default=True,
        description="Whether to enable Gradio's built-in queuing system"
    )
    
    @validator('default_concurrency_limit')
    def validate_concurrency_limit(cls, v):
        """Ensure concurrency limit is reasonable for production use"""
        if v > 20:
            raise ValueError("Concurrency limit should not exceed 20 for stability")
        return v


class EventConcurrencyConfig(BaseModel):
    """Configuration for specific event listener concurrency"""
    
    chat_concurrency_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Concurrency limit for chat message processing"
    )
    
    file_upload_concurrency_limit: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Concurrency limit for file upload processing"
    )
    
    stats_refresh_concurrency_limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Concurrency limit for stats refresh operations"
    )
    
    logs_refresh_concurrency_limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Concurrency limit for logs refresh operations"
    )
    
    # Shared queue IDs for resource management
    llm_provider_queue_id: str = Field(
        default="llm_provider_queue",
        description="Shared queue ID for LLM provider operations"
    )
    
    file_processing_queue_id: str = Field(
        default="file_processing_queue",
        description="Shared queue ID for file processing operations"
    )


class ConcurrencyConfig(BaseModel):
    """Main concurrency configuration container"""
    
    queue: QueueConfig = Field(
        default_factory=QueueConfig,
        description="Queue management configuration"
    )
    
    events: EventConcurrencyConfig = Field(
        default_factory=EventConcurrencyConfig,
        description="Event-specific concurrency configuration"
    )
    
    # Environment-based overrides
    enable_concurrent_processing: bool = Field(
        default=True,
        description="Master switch for concurrent processing"
    )
    
    @classmethod
    def from_env(cls) -> 'ConcurrencyConfig':
        """Create configuration from environment variables"""
        return cls(
            queue=QueueConfig(
                default_concurrency_limit=int(os.getenv('GRADIO_CONCURRENCY_LIMIT', '3')),
                max_threads=int(os.getenv('GRADIO_MAX_THREADS', '100')),
                enable_queue=os.getenv('GRADIO_ENABLE_QUEUE', 'true').lower() == 'true'
            ),
            events=EventConcurrencyConfig(
                chat_concurrency_limit=int(os.getenv('CHAT_CONCURRENCY_LIMIT', '3')),
                file_upload_concurrency_limit=int(os.getenv('FILE_CONCURRENCY_LIMIT', '2')),
                stats_refresh_concurrency_limit=int(os.getenv('STATS_CONCURRENCY_LIMIT', '5')),
                logs_refresh_concurrency_limit=int(os.getenv('LOGS_CONCURRENCY_LIMIT', '5'))
            ),
            enable_concurrent_processing=os.getenv('ENABLE_CONCURRENT_PROCESSING', 'true').lower() == 'true'
        )
    
    def to_gradio_queue_config(self) -> Dict[str, Any]:
        """Convert to Gradio queue configuration format"""
        if not self.enable_concurrent_processing or not self.queue.enable_queue:
            return {}
        
        return {
            'default_concurrency_limit': self.queue.default_concurrency_limit
        }
    
    def get_event_concurrency(self, event_type: str) -> Dict[str, Any]:
        """Get concurrency configuration for specific event type"""
        if not self.enable_concurrent_processing:
            return {'concurrency_limit': 1}
        
        event_configs = {
            'chat': {'concurrency_limit': self.events.chat_concurrency_limit},
            'file_upload': {
                'concurrency_limit': self.events.file_upload_concurrency_limit,
                'concurrency_id': self.events.file_processing_queue_id
            },
            'stats_refresh': {'concurrency_limit': self.events.stats_refresh_concurrency_limit},
            'logs_refresh': {'concurrency_limit': self.events.logs_refresh_concurrency_limit}
        }
        
        return event_configs.get(event_type, {'concurrency_limit': self.queue.default_concurrency_limit})


# Global configuration instance
_concurrency_config: Optional[ConcurrencyConfig] = None


def get_concurrency_config() -> ConcurrencyConfig:
    """Get the global concurrency configuration instance"""
    global _concurrency_config
    if _concurrency_config is None:
        _concurrency_config = ConcurrencyConfig.from_env()
    return _concurrency_config


def set_concurrency_config(config: ConcurrencyConfig) -> None:
    """Set the global concurrency configuration instance"""
    global _concurrency_config
    _concurrency_config = config


def reset_concurrency_config() -> None:
    """Reset the global concurrency configuration to default"""
    global _concurrency_config
    _concurrency_config = None

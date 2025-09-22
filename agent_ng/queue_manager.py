"""
Queue Manager Module
===================

A lean, modular queue manager for Gradio applications that provides
concurrency control and resource management in a Gradio-native way.

Key Features:
- Gradio-native queue configuration
- Resource-aware concurrency management
- Event-specific queue control
- Clean integration with existing app architecture
- Type-safe configuration using Pydantic models
"""

from typing import Dict, Any, Optional, Callable, List
import gradio as gr
from functools import wraps
import logging

try:
    from .concurrency_config import get_concurrency_config, ConcurrencyConfig
except ImportError:
    from concurrency_config import get_concurrency_config, ConcurrencyConfig


class QueueManager:
    """
    Lean queue manager for Gradio applications with concurrency control.
    
    This manager provides a clean interface for configuring Gradio's built-in
    queuing system with proper concurrency limits and resource management.
    """
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        """
        Initialize the queue manager.
        
        Args:
            config: Concurrency configuration. If None, uses global config.
        """
        self.config = config or get_concurrency_config()
        self._queue_configured = False
        self._event_handlers: Dict[str, Callable] = {}
        
    def configure_queue(self, demo: gr.Blocks) -> None:
        """
        Configure Gradio queue with concurrency settings following Gradio best practices.
        
        This method implements the exact patterns from the Gradio queuing documentation:
        https://www.gradio.app/guides/queuing
        
        Args:
            demo: Gradio Blocks instance to configure
        """
        if self._queue_configured:
            return
            
        if not self.config.enable_concurrent_processing:
            logging.info("Concurrent processing disabled - using default queue settings")
            return
            
        # Configure queue using Gradio's recommended approach
        queue_config = self.config.to_gradio_queue_config()
        if queue_config:
            # Apply global queue configuration as per Gradio docs
            demo.queue(**queue_config)
            logging.info(f"Configured Gradio queue with global settings: {queue_config}")
        else:
            logging.info("Using default Gradio queue configuration")
            
        self._queue_configured = True
    
    def get_event_concurrency(self, event_type: str) -> Dict[str, Any]:
        """
        Get concurrency configuration for a specific event type.
        
        Args:
            event_type: Type of event (chat, file_upload, etc.)
            
        Returns:
            Dictionary with concurrency settings for the event
        """
        return self.config.get_event_concurrency(event_type)
    
    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """
        Register an event handler for concurrency management.
        
        Args:
            event_name: Name of the event
            handler: Event handler function
        """
        self._event_handlers[event_name] = handler
    
    def apply_concurrency_to_event(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """
        Apply concurrency settings to an event handler.
        
        Args:
            event_type: Type of event to configure
            **kwargs: Additional event handler arguments
            
        Returns:
            Updated kwargs with concurrency settings
        """
        concurrency_config = self.get_event_concurrency(event_type)
        kwargs.update(concurrency_config)
        return kwargs
    
    def create_concurrent_wrapper(self, event_type: str, handler: Callable) -> Callable:
        """
        Create a wrapper function with concurrency control.
        
        Args:
            event_type: Type of event
            handler: Original handler function
            
        Returns:
            Wrapped handler with concurrency control
        """
        concurrency_config = self.get_event_concurrency(event_type)
        
        @wraps(handler)
        def wrapper(*args, **kwargs):
            # Add concurrency metadata to the handler
            if hasattr(handler, '__concurrency_config__'):
                handler.__concurrency_config__.update(concurrency_config)
            else:
                handler.__concurrency_config__ = concurrency_config
            return handler(*args, **kwargs)
        
        return wrapper
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue configuration status.
        
        Returns:
            Dictionary with queue status information
        """
        return {
            'queue_configured': self._queue_configured,
            'concurrent_processing_enabled': self.config.enable_concurrent_processing,
            'default_concurrency_limit': self.config.queue.default_concurrency_limit,
            'max_threads': self.config.queue.max_threads,
            'registered_handlers': list(self._event_handlers.keys())
        }


def create_queue_manager(config: Optional[ConcurrencyConfig] = None) -> QueueManager:
    """
    Factory function to create a QueueManager instance.
    
    Args:
        config: Optional concurrency configuration
        
    Returns:
        QueueManager instance
    """
    return QueueManager(config)


def apply_concurrency_to_click_event(
    queue_manager: QueueManager,
    event_type: str,
    click_fn: Callable,
    inputs: List[Any],
    outputs: List[Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Apply concurrency settings to a Gradio click event.
    
    Args:
        queue_manager: QueueManager instance
        event_type: Type of event (chat, file_upload, etc.)
        click_fn: Click handler function
        inputs: Input components
        outputs: Output components
        **kwargs: Additional click event arguments
        
    Returns:
        Dictionary with click event configuration including concurrency
    """
    concurrency_config = queue_manager.get_event_concurrency(event_type)
    
    return {
        'fn': click_fn,
        'inputs': inputs,
        'outputs': outputs,
        **concurrency_config,
        **kwargs
    }


def apply_concurrency_to_submit_event(
    queue_manager: QueueManager,
    event_type: str,
    submit_fn: Callable,
    inputs: List[Any],
    outputs: List[Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Apply concurrency settings to a Gradio submit event.
    
    Args:
        queue_manager: QueueManager instance
        event_type: Type of event (chat, file_upload, etc.)
        submit_fn: Submit handler function
        inputs: Input components
        outputs: Output components
        **kwargs: Additional submit event arguments
        
    Returns:
        Dictionary with submit event configuration including concurrency
    """
    concurrency_config = queue_manager.get_event_concurrency(event_type)
    
    return {
        'fn': submit_fn,
        'inputs': inputs,
        'outputs': outputs,
        **concurrency_config,
        **kwargs
    }


def apply_concurrency_to_change_event(
    queue_manager: QueueManager,
    event_type: str,
    change_fn: Callable,
    inputs: List[Any],
    outputs: List[Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Apply concurrency settings to a Gradio change event.
    
    Args:
        queue_manager: QueueManager instance
        event_type: Type of event (stats_refresh, logs_refresh, etc.)
        change_fn: Change handler function
        inputs: Input components
        outputs: Output components
        **kwargs: Additional change event arguments
        
    Returns:
        Dictionary with change event configuration including concurrency
    """
    concurrency_config = queue_manager.get_event_concurrency(event_type)
    
    return {
        'fn': change_fn,
        'inputs': inputs,
        'outputs': outputs,
        **concurrency_config,
        **kwargs
    }

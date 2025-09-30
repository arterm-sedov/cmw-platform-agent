"""
Dead Code: Reset/Cleanup Functions
=================================

These functions were defined but never called anywhere in the codebase.
They can be safely removed unless needed for future debugging or maintenance.

Extracted from:
- agent_ng/langchain_agent.py
- agent_ng/llm_manager.py  
- agent_ng/stats_manager.py
- agent_ng/response_processor.py
- agent_ng/token_counter.py
- agent_ng/trace_manager.py
- agent_ng/debug_streamer.py
"""

# From agent_ng/langchain_agent.py
def reset_agent_ng():
    """Reset the global agent instance"""
    global _agent_ng
    _agent_ng = None

# Alias for backward compatibility
reset_langchain_agent = reset_agent_ng

# From agent_ng/llm_manager.py
def reset_llm_manager():
    """Reset the global LLM manager instance"""
    global _llm_manager
    _llm_manager = None

# From agent_ng/stats_manager.py
def reset_stats_manager():
    """Reset the global stats manager instance"""
    global _stats_manager
    _stats_manager = None

# From agent_ng/response_processor.py
def reset_response_processor():
    """Reset the global response processor instance"""
    global _response_processor
    _response_processor = None

# From agent_ng/token_counter.py
def reset_token_tracker(session_id: str = None) -> None:
    """Reset token tracker for a specific session or all sessions"""
    global _token_trackers
    if session_id is None:
        _token_trackers.clear()
    else:
        _token_trackers.pop(session_id, None)

# From agent_ng/trace_manager.py
def reset_trace_manager(session_id: str = None):
    """Reset trace manager for a specific session or all sessions"""
    global _trace_managers
    if session_id is None:
        _trace_managers.clear()
    else:
        _trace_managers.pop(session_id, None)

# From agent_ng/debug_streamer.py
def cleanup_debug_system():
    """Cleanup the debug system"""
    global _debug_streamers, _log_handlers, _thinking_transparencies, _session_aware_handler

    # Stop all debug streamers
    for streamer in _debug_streamers.values():
        streamer.stop()

    # Clear all instances
    _debug_streamers.clear()
    _log_handlers.clear()
    _thinking_transparencies.clear()
    _session_aware_handler = None

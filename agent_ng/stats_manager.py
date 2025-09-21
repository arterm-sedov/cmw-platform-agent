"""
Stats Manager Module
===================

This module handles all statistics tracking and reporting functionality.
It provides utilities for tracking LLM usage, performance metrics, and system statistics.

Key Features:
- LLM usage tracking and performance metrics
- Conversation statistics and history
- System performance monitoring
- Export and reporting capabilities
- Real-time statistics updates

Usage:
    from stats_manager import StatsManager
    
    stats_mgr = StatsManager()
    stats_mgr.track_llm_usage("gemini", "success")
    stats = stats_mgr.get_comprehensive_stats()
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter


@dataclass
class LLMStats:
    """Statistics for a specific LLM"""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    threshold_passes: int = 0
    submissions: int = 0
    low_submissions: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    last_used: Optional[float] = None
    first_used: Optional[float] = None


@dataclass
class ConversationStats:
    """Statistics for conversations"""
    total_conversations: int = 0
    total_questions: int = 0
    avg_questions_per_conversation: float = 0.0
    total_tool_calls: int = 0
    avg_tool_calls_per_question: float = 0.0
    total_duration: float = 0.0
    avg_duration_per_question: float = 0.0


@dataclass
class SystemStats:
    """System-wide statistics"""
    uptime: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    peak_memory_usage: float = 0.0
    current_memory_usage: float = 0.0


class StatsManager:
    """Manages statistics tracking and reporting"""
    
    def __init__(self):
        self.start_time = time.time()
        self.llm_stats: Dict[str, LLMStats] = {}
        self.conversation_stats = ConversationStats()
        self.system_stats = SystemStats()
        self.llm_tracking: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.conversation_histories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.error_counts: Counter = Counter()
        self.performance_metrics: List[Dict[str, Any]] = []
        
        # Initialize LLM stats for known providers
        self._init_llm_stats()
    
    def _init_llm_stats(self):
        """Initialize LLM stats for known providers"""
        providers = ['gemini', 'groq', 'huggingface', 'openrouter', 'mistral', 'gigachat']
        for provider in providers:
            self.llm_stats[provider] = LLMStats(name=provider)
    
    def track_llm_usage(self, llm_type: str, event_type: str, response_time: float = 0.0, session_id: str = "default"):
        """
        Track LLM usage statistics.
        
        Args:
            llm_type: Type of LLM (e.g., 'gemini', 'groq')
            event_type: Type of event ('success', 'failure', 'threshold_pass', 'submitted', 'low_submit')
            response_time: Response time in seconds
            session_id: Session ID for isolation
        """
        # Use session-specific key for tracking
        session_key = f"{session_id}_{llm_type}"
        if session_key not in self.llm_stats:
            self.llm_stats[session_key] = LLMStats(name=llm_type)
        
        stats = self.llm_stats[session_key]
        current_time = time.time()
        
        # Update session-specific tracking counters
        session_tracking_key = f"{session_id}_{llm_type}"
        self.llm_tracking[session_tracking_key][event_type] += 1
        
        # Update LLM stats
        if event_type == 'success':
            stats.successful_calls += 1
            stats.total_calls += 1
            stats.total_response_time += response_time
            stats.avg_response_time = stats.total_response_time / stats.successful_calls
            stats.last_used = current_time
            if stats.first_used is None:
                stats.first_used = current_time
        elif event_type == 'failure':
            stats.failed_calls += 1
            stats.total_calls += 1
        elif event_type == 'threshold_pass':
            stats.threshold_passes += 1
        elif event_type == 'submitted':
            stats.submissions += 1
        elif event_type == 'low_submit':
            stats.low_submissions += 1
        
        # Track performance metrics
        self.performance_metrics.append({
            'timestamp': current_time,
            'llm_type': llm_type,
            'event_type': event_type,
            'response_time': response_time
        })
        
        # Keep only last 1000 metrics to prevent memory growth
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
    
    def track_conversation(self, conversation_id: str, question: str, answer: str, 
                          llm_used: str, tool_calls: int = 0, duration: float = 0.0, session_id: str = "default"):
        """
        Track conversation statistics.
        
        Args:
            conversation_id: Unique conversation identifier
            question: User question
            answer: AI answer
            llm_used: LLM that was used
            tool_calls: Number of tool calls made
            duration: Duration in seconds
            session_id: Session ID for isolation
        """
        # Update session-specific conversation stats
        session_conv_key = f"{session_id}_{conversation_id}"
        if session_conv_key not in self.conversation_histories:
            self.conversation_histories[session_conv_key] = []
        
        # Store conversation data
        conversation_data = {
            'question': question,
            'answer': answer,
            'llm_used': llm_used,
            'tool_calls': tool_calls,
            'duration': duration,
            'timestamp': time.time()
        }
        self.conversation_histories[session_conv_key].append(conversation_data)
        
        # Update global conversation stats
        self.conversation_stats.total_questions += 1
        self.conversation_stats.total_tool_calls += tool_calls
        self.conversation_stats.total_duration += duration
        
        # Update averages
        if self.conversation_stats.total_questions > 0:
            self.conversation_stats.avg_tool_calls_per_question = (
                self.conversation_stats.total_tool_calls / self.conversation_stats.total_questions
            )
            self.conversation_stats.avg_duration_per_question = (
                self.conversation_stats.total_duration / self.conversation_stats.total_questions
            )
        
        # Update system stats
        self.system_stats.total_requests += 1
        self.system_stats.successful_requests += 1
        self._update_error_rate()
    
    def track_error(self, error_type: str, llm_type: str = None):
        """
        Track error occurrences.
        
        Args:
            error_type: Type of error
            llm_type: LLM type where error occurred
        """
        self.error_counts[error_type] += 1
        self.system_stats.failed_requests += 1
        self._update_error_rate()
        
        if llm_type:
            self.track_llm_usage(llm_type, 'failure')
    
    def _update_error_rate(self):
        """Update error rate calculation"""
        total = self.system_stats.successful_requests + self.system_stats.failed_requests
        if total > 0:
            self.system_stats.error_rate = self.system_stats.failed_requests / total
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """Get LLM statistics"""
        stats = {}
        for name, llm_stats in self.llm_stats.items():
            stats[name] = {
                'name': llm_stats.name,
                'total_calls': llm_stats.total_calls,
                'successful_calls': llm_stats.successful_calls,
                'failed_calls': llm_stats.failed_calls,
                'success_rate': (
                    llm_stats.successful_calls / llm_stats.total_calls 
                    if llm_stats.total_calls > 0 else 0
                ),
                'threshold_passes': llm_stats.threshold_passes,
                'submissions': llm_stats.submissions,
                'low_submissions': llm_stats.low_submissions,
                'avg_response_time': llm_stats.avg_response_time,
                'last_used': llm_stats.last_used,
                'first_used': llm_stats.first_used
            }
        return stats
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            'total_conversations': len(self.conversation_histories),
            'total_questions': self.conversation_stats.total_questions,
            'avg_questions_per_conversation': (
                self.conversation_stats.total_questions / len(self.conversation_histories)
                if self.conversation_histories else 0
            ),
            'total_tool_calls': self.conversation_stats.total_tool_calls,
            'avg_tool_calls_per_question': self.conversation_stats.avg_tool_calls_per_question,
            'total_duration': self.conversation_stats.total_duration,
            'avg_duration_per_question': self.conversation_stats.avg_duration_per_question
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            'uptime': uptime,
            'total_requests': self.system_stats.total_requests,
            'successful_requests': self.system_stats.successful_requests,
            'failed_requests': self.system_stats.failed_requests,
            'error_rate': self.system_stats.error_rate,
            'requests_per_minute': (
                self.system_stats.total_requests / (uptime / 60) if uptime > 0 else 0
            ),
            'error_counts': dict(self.error_counts),
            'performance_metrics_count': len(self.performance_metrics)
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'llm_stats': self.get_llm_stats(),
            'conversation_stats': self.get_conversation_stats(),
            'system_stats': self.get_system_stats(),
            'timestamp': time.time()
        }
    
    def get_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """Get session-specific statistics"""
        return self.get_session_stats(session_id)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        # Filter stats by session
        session_llm_stats = {}
        for key, stats in self.llm_stats.items():
            if key.startswith(f"{session_id}_"):
                llm_type = key.replace(f"{session_id}_", "")
                session_llm_stats[llm_type] = stats
        
        # Count session-specific conversations
        session_conversations = 0
        for key in self.conversation_histories.keys():
            if key.startswith(f"{session_id}_"):
                session_conversations += len(self.conversation_histories[key])
        
        return {
            'llm_stats': session_llm_stats,
            'conversation_count': session_conversations,
            'system_stats': self.get_system_stats()  # System stats are global
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics (all sessions)"""
        return {
            'llm_stats': self.get_llm_stats(),
            'conversation_stats': self.get_conversation_stats(),
            'system_stats': self.get_system_stats(),
        }
    
    def get_llm_stats_json(self) -> str:
        """Get LLM statistics as JSON string"""
        stats = self.get_llm_stats()
        return json.dumps(stats, indent=2)
    
    def get_performance_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance metrics"""
        return self.performance_metrics[-limit:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_types': dict(self.error_counts),
            'most_common_error': self.error_counts.most_common(1)[0] if total_errors > 0 else None
        }
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation"""
        return self.conversation_histories.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a specific conversation"""
        if conversation_id in self.conversation_histories:
            del self.conversation_histories[conversation_id]
    
    def clear_all_stats(self):
        """Clear all statistics"""
        self.llm_stats.clear()
        self.conversation_histories.clear()
        self.error_counts.clear()
        self.performance_metrics.clear()
        self.llm_tracking.clear()
        self._init_llm_stats()
        self.conversation_stats = ConversationStats()
        self.system_stats = SystemStats()
        self.start_time = time.time()
    
    def export_stats(self, filename: str = None) -> str:
        """Export statistics to JSON file"""
        if filename is None:
            filename = f"stats_export_{int(time.time())}.json"
        
        stats = self.get_comprehensive_stats()
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return filename
    
    def get_stats_summary(self) -> str:
        """Get a human-readable stats summary"""
        llm_stats = self.get_llm_stats()
        conversation_stats = self.get_conversation_stats()
        system_stats = self.get_system_stats()
        
        summary = f"""
=== STATISTICS SUMMARY ===
System Uptime: {system_stats['uptime']:.2f} seconds
Total Requests: {system_stats['total_requests']}
Success Rate: {(1 - system_stats['error_rate']) * 100:.1f}%
Requests/Minute: {system_stats['requests_per_minute']:.2f}

Conversation Stats:
- Total Questions: {conversation_stats['total_questions']}
- Total Tool Calls: {conversation_stats['total_tool_calls']}
- Avg Tool Calls/Question: {conversation_stats['avg_tool_calls_per_question']:.2f}

LLM Performance:
"""
        
        for name, stats in llm_stats.items():
            if stats['total_calls'] > 0:
                summary += f"- {name}: {stats['success_rate']*100:.1f}% success ({stats['total_calls']} calls)\n"
        
        return summary


# Global stats manager instance
_stats_manager = None

def get_stats_manager() -> StatsManager:
    """Get the global stats manager instance"""
    global _stats_manager
    if _stats_manager is None:
        _stats_manager = StatsManager()
    return _stats_manager

def reset_stats_manager():
    """Reset the global stats manager instance"""
    global _stats_manager
    _stats_manager = None

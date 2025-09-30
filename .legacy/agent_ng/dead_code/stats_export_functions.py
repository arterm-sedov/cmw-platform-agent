"""
Dead Code: Stats/Export Functions
================================

These statistics and export functions were defined but never called anywhere in the codebase.
They can be safely removed unless needed for future analytics or reporting.

Extracted from:
- agent_ng/stats_manager.py
"""

# From agent_ng/stats_manager.py
def export_stats(self, filename: str = None) -> str:
    """Export statistics to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_stats_{timestamp}.json"
    
    stats_data = self.get_comprehensive_stats()
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, default=str)
        return f"Stats exported to {filename}"
    except Exception as e:
        return f"Error exporting stats: {e}"

def get_stats_summary(self) -> str:
    """Get a human-readable summary of statistics"""
    stats = self.get_comprehensive_stats()
    
    summary = []
    summary.append("=== Agent Statistics Summary ===")
    summary.append(f"Uptime: {stats['system_stats']['uptime']:.2f} seconds")
    summary.append(f"Total Requests: {stats['system_stats']['total_requests']}")
    summary.append(f"Successful Requests: {stats['system_stats']['successful_requests']}")
    summary.append(f"Failed Requests: {stats['system_stats']['failed_requests']}")
    summary.append(f"Error Rate: {stats['system_stats']['error_rate']:.2%}")
    
    # LLM Stats
    summary.append("\\n=== LLM Usage ===")
    for llm_name, llm_stats in stats['llm_stats'].items():
        summary.append(f"{llm_name}:")
        summary.append(f"  Total Calls: {llm_stats['total_calls']}")
        summary.append(f"  Successful: {llm_stats['successful_calls']}")
        summary.append(f"  Failed: {llm_stats['failed_calls']}")
        summary.append(f"  Avg Response Time: {llm_stats['avg_response_time']:.2f}s")
    
    # Conversation Stats
    summary.append("\\n=== Conversation Stats ===")
    conv_stats = stats['conversation_stats']
    summary.append(f"Total Conversations: {conv_stats['total_conversations']}")
    summary.append(f"Total Questions: {conv_stats['total_questions']}")
    summary.append(f"Avg Questions per Conversation: {conv_stats['avg_questions_per_conversation']:.2f}")
    summary.append(f"Total Tool Calls: {conv_stats['total_tool_calls']}")
    summary.append(f"Avg Tool Calls per Question: {conv_stats['avg_tool_calls_per_question']:.2f}")
    
    return "\\n".join(summary)

def get_performance_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent performance metrics"""
    return self.performance_metrics[-limit:]

def get_error_summary(self) -> Dict[str, Any]:
    """Get error summary statistics"""
    error_counts = {}
    for metric in self.performance_metrics:
        if 'error_type' in metric:
            error_type = metric['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    return {
        'total_errors': sum(error_counts.values()),
        'error_breakdown': error_counts,
        'most_common_error': max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
    }

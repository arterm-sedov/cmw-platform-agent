"""
Trace Management Module
======================

This module handles all tracing, logging, and debug output capture functionality.
It provides decorators, classes, and methods for comprehensive execution tracing.

Key Features:
- Print statement tracing with context
- Debug output capture and buffering
- Trace data serialization and management
- Streaming event tracking
- LLM call tracing and monitoring

Usage:
    from trace_manager import trace_prints_with_context, trace_prints, TraceManager
    
    @trace_prints_with_context("tool_execution")
    def my_function(self):
        print("This will be traced")
"""

import sys
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from io import StringIO
from contextlib import contextmanager


@dataclass
class TraceEvent:
    """Represents a single trace event"""
    timestamp: float
    event_type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    call_id: Optional[str] = None


@dataclass
class LLMCallTrace:
    """Represents an LLM call trace"""
    call_id: str
    llm_type: str
    input_messages: List[Any]
    output_response: Any
    execution_time: float
    timestamp: float
    use_tools: bool = False
    error: Optional[str] = None


@dataclass
class QuestionTrace:
    """Represents a complete question trace"""
    question: str
    file_data: Optional[str] = None
    file_name: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    llm_calls: List[LLMCallTrace] = None
    final_answer: Optional[str] = None
    llm_used: Optional[str] = None
    reference: Optional[str] = None
    debug_output: List[str] = None
    
    def __post_init__(self):
        if self.llm_calls is None:
            self.llm_calls = []
        if self.debug_output is None:
            self.debug_output = []


def trace_prints_with_context(context_type: str):
    """
    Decorator that traces all print calls in a function and attaches them to specific execution contexts.
    Automatically captures print output and adds it to the appropriate context in the agent's trace.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Store original print
            original_print = print
            
            # Store current context
            old_context = getattr(self, '_current_trace_context', None)
            self._current_trace_context = context_type
            
            def trace_print(*print_args, **print_kwargs):
                # Original print functionality
                original_print(*print_args, **print_kwargs)
                
                # Write to current LLM's stdout buffer if available
                if hasattr(self, 'current_llm_stdout_buffer') and self.current_llm_stdout_buffer:
                    try:
                        message = " ".join(str(arg) for arg in print_args)
                        self.current_llm_stdout_buffer.write(message + "\n")
                    except Exception as e:
                        # Fallback if buffer write fails
                        original_print(f"[Buffer Error] Failed to write to stdout buffer: {e}")
                
                # Add to trace if trace manager is available
                if hasattr(self, 'trace_manager'):
                    try:
                        message = " ".join(str(arg) for arg in print_args)
                        self.trace_manager.add_debug_output(message, context_type)
                    except Exception as e:
                        # Fallback if trace manager fails
                        original_print(f"[Trace Error] Failed to add debug output: {e}")
            
            # Replace print with traced version
            print = trace_print
            
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Restore original print
                print = original_print
                # Restore old context
                self._current_trace_context = old_context
        
        return wrapper
    return decorator


def trace_prints(func):
    """
    Decorator that traces all print calls in a function.
    Automatically captures print output and adds it to the agent's trace.
    """
    def wrapper(self, *args, **kwargs):
        # Store original print
        original_print = print
        
        def trace_print(*print_args, **print_kwargs):
            # Original print functionality
            original_print(*print_args, **print_kwargs)
            
            # Write to current LLM's stdout buffer if available
            if hasattr(self, 'current_llm_stdout_buffer') and self.current_llm_stdout_buffer:
                try:
                    message = " ".join(str(arg) for arg in print_args)
                    self.current_llm_stdout_buffer.write(message + "\n")
                except Exception as e:
                    # Fallback if buffer write fails
                    original_print(f"[Buffer Error] Failed to write to stdout buffer: {e}")
            
            # Add to trace if trace manager is available
            if hasattr(self, 'trace_manager'):
                try:
                    message = " ".join(str(arg) for arg in print_args)
                    context = getattr(self, '_current_trace_context', 'general')
                    self.trace_manager.add_debug_output(message, context)
                except Exception as e:
                    # Fallback if trace manager fails
                    original_print(f"[Trace Error] Failed to add debug output: {e}")
        
        # Replace print with traced version
        print = trace_print
        
        try:
            result = func(self, *args, **kwargs)
            return result
        finally:
            # Restore original print
            print = original_print
    
    return wrapper


class Tee:
    """
    Tee class to duplicate writes to multiple streams (e.g., sys.stdout and a buffer).
    """
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for s in self.streams:
            s.write(data)
    
    def flush(self):
        for s in self.streams:
            s.flush()


class _SinkWriter:
    """Writer that sends data to a sink function"""
    def __init__(self, sink):
        self.sink = sink
    
    def write(self, data):
        try:
            self.sink(data)
        except Exception:
            pass
    
    def flush(self):
        pass


class TraceManager:
    """Manages all tracing functionality for the agent"""
    
    def __init__(self):
        self.current_trace: Optional[QuestionTrace] = None
        self.trace_history: List[QuestionTrace] = []
        self.debug_output: List[str] = []
        self.current_llm_stdout_buffer: Optional[StringIO] = None
        self._current_trace_context: Optional[str] = None
    
    def init_question(self, question: str, file_data: str = None, file_name: str = None):
        """Initialize a new question trace"""
        self.current_trace = QuestionTrace(
            question=question,
            file_data=file_data,
            file_name=file_name,
            start_time=time.time()
        )
        self.debug_output = []
        self.current_llm_stdout_buffer = StringIO()
    
    def start_llm(self, llm_type: str) -> str:
        """Start tracing an LLM call"""
        call_id = str(uuid.uuid4())
        if self.current_trace:
            self.current_trace.llm_used = llm_type
        return call_id
    
    def capture_llm_stdout(self, llm_type: str, call_id: str):
        """Capture stdout for an LLM call"""
        if self.current_llm_stdout_buffer:
            try:
                output = self.current_llm_stdout_buffer.getvalue()
                if output.strip():
                    self.add_debug_output(f"[{llm_type}] {output.strip()}", "llm_stdout")
                self.current_llm_stdout_buffer = StringIO()
            except Exception as e:
                print(f"[Trace Error] Failed to capture LLM stdout: {e}")
    
    def add_llm_call_input(self, llm_type: str, call_id: str, messages: List, use_tools: bool):
        """Add LLM call input to trace"""
        if self.current_trace:
            llm_call = LLMCallTrace(
                call_id=call_id,
                llm_type=llm_type,
                input_messages=messages,
                output_response=None,
                execution_time=0.0,
                timestamp=time.time(),
                use_tools=use_tools
            )
            self.current_trace.llm_calls.append(llm_call)
    
    def add_llm_call_output(self, llm_type: str, call_id: str, response: Any, execution_time: float):
        """Add LLM call output to trace"""
        if self.current_trace:
            for llm_call in self.current_trace.llm_calls:
                if llm_call.call_id == call_id:
                    llm_call.output_response = response
                    llm_call.execution_time = execution_time
                    break
    
    def add_llm_error(self, llm_type: str, call_id: str, error: Exception):
        """Add LLM error to trace"""
        if self.current_trace:
            for llm_call in self.current_trace.llm_calls:
                if llm_call.call_id == call_id:
                    llm_call.error = str(error)
                    break
    
    def add_debug_output(self, message: str, context: str = "general"):
        """Add debug output to trace"""
        self.debug_output.append(f"[{context}] {message}")
        if self.current_trace:
            self.current_trace.debug_output.append(f"[{context}] {message}")
    
    def finalize_question(self, final_result: dict):
        """Finalize the current question trace"""
        if self.current_trace:
            self.current_trace.end_time = time.time()
            self.current_trace.final_answer = final_result.get('answer', '')
            self.current_trace.llm_used = final_result.get('llm_used', '')
            self.current_trace.reference = final_result.get('reference', '')
            
            # Add to history
            self.trace_history.append(self.current_trace)
            
            # Clear current trace
            self.current_trace = None
            self.debug_output = []
            self.current_llm_stdout_buffer = None
    
    def get_full_trace(self) -> dict:
        """Get the complete trace data"""
        if not self.current_trace:
            return {"error": "No active trace"}
        
        return {
            "question": self.current_trace.question,
            "file_data": self.current_trace.file_data,
            "file_name": self.current_trace.file_name,
            "start_time": self.current_trace.start_time,
            "end_time": self.current_trace.end_time,
            "duration": self.current_trace.end_time - self.current_trace.start_time if self.current_trace.end_time else 0,
            "llm_used": self.current_trace.llm_used,
            "final_answer": self.current_trace.final_answer,
            "reference": self.current_trace.reference,
            "llm_calls": [self._serialize_trace_data(call) for call in self.current_trace.llm_calls],
            "debug_output": self.current_trace.debug_output
        }
    
    def _serialize_trace_data(self, obj):
        """Serialize trace data for JSON compatibility"""
        if hasattr(obj, '__dict__'):
            return {k: self._serialize_trace_data(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._serialize_trace_data(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_trace_data(v) for k, v in obj.items()}
        else:
            return obj
    
    def clear_trace(self):
        """Clear current trace"""
        self.current_trace = None
        self.debug_output = []
        self.current_llm_stdout_buffer = None
    
    def get_trace_history(self) -> List[dict]:
        """Get trace history"""
        return [self._serialize_trace_data(trace) for trace in self.trace_history]
    
    def get_stats(self) -> dict:
        """Get trace statistics"""
        if not self.trace_history:
            return {"total_questions": 0}
        
        total_questions = len(self.trace_history)
        total_llm_calls = sum(len(trace.llm_calls) for trace in self.trace_history)
        avg_duration = sum(
            (trace.end_time - trace.start_time) for trace in self.trace_history 
            if trace.end_time and trace.start_time
        ) / total_questions if total_questions > 0 else 0
        
        return {
            "total_questions": total_questions,
            "total_llm_calls": total_llm_calls,
            "avg_duration": avg_duration,
            "current_trace_active": self.current_trace is not None
        }


# Global trace manager instance
_trace_manager = None

def get_trace_manager() -> TraceManager:
    """Get the global trace manager instance"""
    global _trace_manager
    if _trace_manager is None:
        _trace_manager = TraceManager()
    return _trace_manager

def reset_trace_manager():
    """Reset the global trace manager instance"""
    global _trace_manager
    _trace_manager = None

# Phase 1: Modularize Core Agent - Detailed Implementation Plan

## Executive Summary

**Objective**: Add modular components to the Core Agent while preserving all existing functionality and interfaces.

**Duration**: Week 1 (5 days)
**Risk Level**: **LOW** - No interface changes, only internal refactoring
**Success Criteria**: Core Agent continues to work exactly as before, but with modular architecture

## Current State Analysis

### **Core Agent Current Architecture**
```python
# agent_ng/core_agent.py
class CoreAgent:
    def __init__(self):
        # Only 2 modular components
        self.llm_manager = get_llm_manager()
        self.error_handler = get_error_handler()
        
        # Monolithic components
        self.conversations = defaultdict(list)  # Memory management
        self.conversation_metadata = defaultdict(dict)
        self.current_question = None
        self.current_file_data = None
        self.current_file_name = None
        # ... other monolithic code
```

### **Target Architecture**
```python
# agent_ng/core_agent.py (after Phase 1)
class CoreAgent:
    def __init__(self):
        # Existing working components
        self.llm_manager = get_llm_manager()
        self.error_handler = get_error_handler()
        
        # NEW: Add missing modular components
        self.memory_manager = get_memory_manager()
        self.streaming_manager = get_streaming_manager()
        self.message_processor = get_message_processor()
        self.response_processor = get_response_processor()
        self.stats_manager = get_stats_manager()
        self.trace_manager = get_trace_manager()
```

## Implementation Plan

### **Day 1: Setup and Memory Manager Integration**

#### **Task 1.1: Create Memory Manager Module**
**File**: `agent_ng/memory_manager.py`
**Duration**: 2 hours

```python
"""
Memory Manager for Core Agent
============================

Provides modular memory management for the Core Agent while maintaining
compatibility with existing conversation storage patterns.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from threading import Lock
from collections import defaultdict

@dataclass
class ConversationMessage:
    """Represents a conversation message with metadata"""
    role: str
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class CoreAgentMemoryManager:
    """Memory manager for Core Agent with backward compatibility"""
    
    def __init__(self):
        # Maintain existing data structures for compatibility
        self.conversations: Dict[str, List[ConversationMessage]] = defaultdict(list)
        self.conversation_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.conversation_lock = Lock()
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to conversation history"""
        with self.conversation_lock:
            message = ConversationMessage(
                role=role,
                content=content,
                timestamp=time.time(),
                metadata=metadata
            )
            self.conversations[conversation_id].append(message)
    
    def get_conversation(self, conversation_id: str) -> List[ConversationMessage]:
        """Get conversation history"""
        with self.conversation_lock:
            return self.conversations[conversation_id].copy()
    
    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history"""
        with self.conversation_lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation statistics"""
        messages = self.get_conversation(conversation_id)
        return {
            "message_count": len(messages),
            "user_messages": len([m for m in messages if m.role == "user"]),
            "assistant_messages": len([m for m in messages if m.role == "assistant"]),
            "last_message_time": messages[-1].timestamp if messages else None
        }

# Global instance
_memory_manager = None

def get_memory_manager() -> CoreAgentMemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = CoreAgentMemoryManager()
    return _memory_manager
```

#### **Task 1.2: Integrate Memory Manager into Core Agent**
**File**: `agent_ng/core_agent.py`
**Duration**: 1 hour

```python
# Add to imports
from .memory_manager import get_memory_manager

# Modify __init__ method
def __init__(self):
    # Existing components
    self.llm_manager = get_llm_manager()
    self.error_handler = get_error_handler()
    
    # NEW: Add memory manager
    self.memory_manager = get_memory_manager()
    
    # Keep existing state for compatibility
    self.current_question = None
    self.current_file_data = None
    self.current_file_name = None
    self.total_questions = 0
    
    # Load system prompt
    self.system_prompt = self._load_system_prompt()
    self.sys_msg = SystemMessage(content=self.system_prompt)
    
    # Initialize tools
    self.tools = self._initialize_tools()
```

#### **Task 1.3: Update Memory-Related Methods**
**Duration**: 2 hours

Replace existing memory management with memory manager calls:

```python
# Replace this:
def _add_to_conversation(self, conversation_id: str, role: str, content: str, 
                       metadata: Optional[Dict[str, Any]] = None, 
                       tool_calls: Optional[List[Dict[str, Any]]] = None):
    with self.conversation_lock:
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )
        self.conversations[conversation_id].append(message)

# With this:
def _add_to_conversation(self, conversation_id: str, role: str, content: str, 
                       metadata: Optional[Dict[str, Any]] = None, 
                       tool_calls: Optional[List[Dict[str, Any]]] = None):
    self.memory_manager.add_message(conversation_id, role, content, metadata)
```

### **Day 2: Streaming Manager Integration**

#### **Task 2.1: Create Streaming Manager Module**
**File**: `agent_ng/streaming_manager.py`
**Duration**: 3 hours

```python
"""
Streaming Manager for Core Agent
===============================

Provides modular streaming functionality for the Core Agent while maintaining
compatibility with existing streaming patterns.
"""

from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
import time

class CoreAgentStreamingManager:
    """Streaming manager for Core Agent with backward compatibility"""
    
    def __init__(self):
        self.streaming_callbacks = []
    
    def add_callback(self, callback):
        """Add a streaming callback"""
        self.streaming_callbacks.append(callback)
    
    def remove_callback(self, callback):
        """Remove a streaming callback"""
        if callback in self.streaming_callbacks:
            self.streaming_callbacks.remove(callback)
    
    async def stream_thinking(self, question: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream thinking process"""
        yield {
            "event_type": "thinking",
            "content": f"🤔 Processing: {question[:50]}...",
            "timestamp": time.time()
        }
        
        # Simulate thinking process
        await asyncio.sleep(0.1)
        yield {
            "event_type": "thinking",
            "content": "🧠 Analyzing question and determining approach...",
            "timestamp": time.time()
        }
    
    async def stream_tool_execution(self, tool_name: str, tool_args: dict) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream tool execution"""
        yield {
            "event_type": "tool_start",
            "content": f"🔧 Executing tool: {tool_name}",
            "metadata": {"tool_name": tool_name, "tool_args": tool_args},
            "timestamp": time.time()
        }
        
        # Simulate tool execution
        await asyncio.sleep(0.2)
        
        yield {
            "event_type": "tool_progress",
            "content": f"⚙️ {tool_name} in progress...",
            "metadata": {"tool_name": tool_name},
            "timestamp": time.time()
        }
        
        yield {
            "event_type": "tool_end",
            "content": f"✅ {tool_name} completed",
            "metadata": {"tool_name": tool_name},
            "timestamp": time.time()
        }
    
    async def stream_response(self, response: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response content character by character"""
        for char in response:
            yield {
                "event_type": "content",
                "content": char,
                "timestamp": time.time()
            }
            await asyncio.sleep(0.01)  # Small delay for streaming effect

# Global instance
_streaming_manager = None

def get_streaming_manager() -> CoreAgentStreamingManager:
    """Get the global streaming manager instance"""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = CoreAgentStreamingManager()
    return _streaming_manager
```

#### **Task 2.2: Integrate Streaming Manager into Core Agent**
**Duration**: 2 hours

```python
# Add to imports
from .streaming_manager import get_streaming_manager

# Modify __init__ method
def __init__(self):
    # ... existing code ...
    
    # NEW: Add streaming manager
    self.streaming_manager = get_streaming_manager()
```

#### **Task 2.3: Update Streaming Methods**
**Duration**: 2 hours

Enhance existing `process_question_stream()` method:

```python
async def process_question_stream(self, question: str, file_data: str = None, 
                                file_name: str = None, llm_sequence: Optional[List[str]] = None,
                                chat_history: Optional[List[Dict[str, Any]]] = None,
                                conversation_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
    """Enhanced streaming with modular components"""
    
    # Use streaming manager for thinking process
    async for event in self.streaming_manager.stream_thinking(question):
        yield event
    
    # ... existing code ...
    
    # Use streaming manager for tool execution
    if tool_calls:
        for tool_call in tool_calls:
            async for event in self.streaming_manager.stream_tool_execution(
                tool_call['name'], tool_call['args']
            ):
                yield event
    
    # Use streaming manager for response
    async for event in self.streaming_manager.stream_response(response):
        yield event
```

### **Day 3: Message and Response Processors**

#### **Task 3.1: Create Message Processor Module**
**File**: `agent_ng/message_processor.py`
**Duration**: 2 hours

```python
"""
Message Processor for Core Agent
===============================

Handles message formatting, validation, and preprocessing for the Core Agent.
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class CoreAgentMessageProcessor:
    """Message processor for Core Agent"""
    
    def __init__(self):
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        # ... existing system prompt loading logic
        return "You are a helpful AI assistant."
    
    def format_messages(self, question: str, file_data: str = None, 
                       file_name: str = None, 
                       chat_history: Optional[List[Dict[str, Any]]] = None) -> List[BaseMessage]:
        """Format messages for LLM processing"""
        messages = []
        
        # Add system message
        messages.append(SystemMessage(content=self.system_prompt))
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
        
        # Add current question
        question_text = question
        if file_data and file_name:
            question_text += f"\n\nFile: {file_name}\nContent: {file_data}"
        
        messages.append(HumanMessage(content=question_text))
        
        return messages
    
    def validate_message(self, message: str) -> bool:
        """Validate message before processing"""
        if not message or not message.strip():
            return False
        if len(message) > 10000:  # Reasonable limit
            return False
        return True

# Global instance
_message_processor = None

def get_message_processor() -> CoreAgentMessageProcessor:
    """Get the global message processor instance"""
    global _message_processor
    if _message_processor is None:
        _message_processor = CoreAgentMessageProcessor()
    return _message_processor
```

#### **Task 3.2: Create Response Processor Module**
**File**: `agent_ng/response_processor.py`
**Duration**: 2 hours

```python
"""
Response Processor for Core Agent
================================

Handles response formatting, validation, and post-processing for the Core Agent.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class AgentResponse:
    """Agent response with metadata"""
    answer: str
    tool_calls: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class CoreAgentResponseProcessor:
    """Response processor for Core Agent"""
    
    def __init__(self):
        self.response_templates = {
            "success": "✅ {answer}",
            "error": "❌ Error: {error}",
            "thinking": "🤔 {content}"
        }
    
    def process_response(self, answer: str, tool_calls: List[Dict[str, Any]] = None, 
                        success: bool = True, error: str = None) -> AgentResponse:
        """Process and format agent response"""
        return AgentResponse(
            answer=answer,
            tool_calls=tool_calls or [],
            success=success,
            error=error,
            metadata={"timestamp": time.time()}
        )
    
    def format_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Format tool calls for display"""
        if not tool_calls:
            return ""
        
        formatted = "\n\n🔧 **Tools Used:**\n"
        for i, tool_call in enumerate(tool_calls, 1):
            formatted += f"{i}. **{tool_call.get('name', 'Unknown')}**\n"
            if tool_call.get('args'):
                formatted += f"   Arguments: {tool_call['args']}\n"
        
        return formatted
    
    def validate_response(self, response: AgentResponse) -> bool:
        """Validate agent response"""
        if not response.success and not response.error:
            return False
        if response.success and not response.answer:
            return False
        return True

# Global instance
_response_processor = None

def get_response_processor() -> CoreAgentResponseProcessor:
    """Get the global response processor instance"""
    global _response_processor
    if _response_processor is None:
        _response_processor = CoreAgentResponseProcessor()
    return _response_processor
```

#### **Task 3.3: Integrate Processors into Core Agent**
**Duration**: 1 hour

```python
# Add to imports
from .message_processor import get_message_processor
from .response_processor import get_response_processor

# Modify __init__ method
def __init__(self):
    # ... existing code ...
    
    # NEW: Add processors
    self.message_processor = get_message_processor()
    self.response_processor = get_response_processor()
```

### **Day 4: Statistics and Trace Managers**

#### **Task 4.1: Enhance Statistics Manager**
**File**: `agent_ng/stats_manager.py` (enhance existing)
**Duration**: 2 hours

```python
"""
Enhanced Statistics Manager for Core Agent
=========================================

Provides comprehensive statistics and monitoring for the Core Agent.
"""

class CoreAgentStatsManager:
    """Enhanced statistics manager for Core Agent"""
    
    def __init__(self):
        self.stats = {
            "total_questions": 0,
            "successful_questions": 0,
            "failed_questions": 0,
            "total_tool_calls": 0,
            "conversation_count": 0,
            "start_time": time.time()
        }
        self.conversation_stats = {}
    
    def increment_questions(self, success: bool = True):
        """Increment question counter"""
        self.stats["total_questions"] += 1
        if success:
            self.stats["successful_questions"] += 1
        else:
            self.stats["failed_questions"] += 1
    
    def increment_tool_calls(self, count: int = 1):
        """Increment tool call counter"""
        self.stats["total_tool_calls"] += count
    
    def update_conversation_stats(self, conversation_id: str, stats: Dict[str, Any]):
        """Update conversation-specific statistics"""
        self.conversation_stats[conversation_id] = stats
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get overall agent statistics"""
        return {
            **self.stats,
            "uptime": time.time() - self.stats["start_time"],
            "success_rate": (
                self.stats["successful_questions"] / self.stats["total_questions"]
                if self.stats["total_questions"] > 0 else 0
            )
        }
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation-specific statistics"""
        return self.conversation_stats.get(conversation_id, {})
```

#### **Task 4.2: Create Trace Manager Module**
**File**: `agent_ng/trace_manager.py` (enhance existing)
**Duration**: 2 hours

```python
"""
Enhanced Trace Manager for Core Agent
====================================

Provides comprehensive tracing and debugging for the Core Agent.
"""

class CoreAgentTraceManager:
    """Enhanced trace manager for Core Agent"""
    
    def __init__(self):
        self.traces = []
        self.current_trace = None
    
    def start_trace(self, question: str, file_data: str = None, file_name: str = None):
        """Start a new trace"""
        self.current_trace = {
            "question": question,
            "file_data": file_data,
            "file_name": file_name,
            "start_time": time.time(),
            "events": []
        }
    
    def add_event(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        """Add an event to current trace"""
        if self.current_trace:
            self.current_trace["events"].append({
                "type": event_type,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time()
            })
    
    def end_trace(self, success: bool = True, error: str = None):
        """End current trace"""
        if self.current_trace:
            self.current_trace["end_time"] = time.time()
            self.current_trace["success"] = success
            self.current_trace["error"] = error
            self.traces.append(self.current_trace)
            self.current_trace = None
    
    def get_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces"""
        return self.traces[-limit:]
```

#### **Task 4.3: Integrate Managers into Core Agent**
**Duration**: 1 hour

```python
# Add to imports
from .stats_manager import get_stats_manager
from .trace_manager import get_trace_manager

# Modify __init__ method
def __init__(self):
    # ... existing code ...
    
    # NEW: Add managers
    self.stats_manager = get_stats_manager()
    self.trace_manager = get_trace_manager()
```

### **Day 5: Integration and Testing**

#### **Task 5.1: Update Core Agent Methods**
**Duration**: 3 hours

Update all methods to use the new modular components:

```python
def process_question(self, question: str, file_data: str = None, file_name: str = None,
                    llm_sequence: Optional[List[str]] = None, 
                    chat_history: Optional[List[Dict[str, Any]]] = None,
                    conversation_id: str = "default") -> AgentResponse:
    """Enhanced process_question with modular components"""
    
    # Start trace
    self.trace_manager.start_trace(question, file_data, file_name)
    
    try:
        # Validate message
        if not self.message_processor.validate_message(question):
            return self.response_processor.process_response(
                "", [], False, "Invalid message"
            )
        
        # Format messages
        messages = self.message_processor.format_messages(
            question, file_data, file_name, chat_history
        )
        
        # Process with LLM
        answer, tool_calls = self._process_with_llm(messages, conversation_id)
        
        # Update statistics
        self.stats_manager.increment_questions(True)
        self.stats_manager.increment_tool_calls(len(tool_calls))
        
        # End trace
        self.trace_manager.end_trace(True)
        
        # Return processed response
        return self.response_processor.process_response(answer, tool_calls, True)
        
    except Exception as e:
        # Update statistics
        self.stats_manager.increment_questions(False)
        
        # End trace
        self.trace_manager.end_trace(False, str(e))
        
        # Return error response
        return self.response_processor.process_response("", [], False, str(e))
```

#### **Task 5.2: Create Integration Tests**
**File**: `misc_files/test_phase_1_integration.py`
**Duration**: 2 hours

```python
"""
Phase 1 Integration Tests
========================

Test that Core Agent works exactly as before with modular components.
"""

import pytest
from agent_ng.core_agent import get_agent

def test_core_agent_interface_compatibility():
    """Test that Core Agent interface remains unchanged"""
    agent = get_agent()
    
    # Test process_question method
    response = agent.process_question("Hello, how are you?")
    assert hasattr(response, 'answer')
    assert hasattr(response, 'tool_calls')
    assert hasattr(response, 'success')
    assert hasattr(response, 'error')

def test_memory_manager_integration():
    """Test that memory manager works correctly"""
    agent = get_agent()
    
    # Test conversation management
    agent._add_to_conversation("test", "user", "Hello")
    history = agent.memory_manager.get_conversation("test")
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "Hello"

def test_streaming_manager_integration():
    """Test that streaming manager works correctly"""
    agent = get_agent()
    
    # Test streaming functionality
    import asyncio
    async def test_streaming():
        events = []
        async for event in agent.streaming_manager.stream_thinking("Test question"):
            events.append(event)
        assert len(events) > 0
        assert events[0]["event_type"] == "thinking"
    
    asyncio.run(test_streaming())

def test_statistics_tracking():
    """Test that statistics are tracked correctly"""
    agent = get_agent()
    
    # Process a question
    response = agent.process_question("Test question")
    
    # Check statistics
    stats = agent.stats_manager.get_agent_stats()
    assert stats["total_questions"] > 0
    assert "success_rate" in stats

def test_trace_manager_integration():
    """Test that trace manager works correctly"""
    agent = get_agent()
    
    # Process a question
    response = agent.process_question("Test question")
    
    # Check traces
    traces = agent.trace_manager.get_traces()
    assert len(traces) > 0
    assert traces[-1]["question"] == "Test question"
```

#### **Task 5.3: Update Documentation**
**Duration**: 1 hour

Update `docs/CORE_AGENT_VS_LANGCHAIN_AGENT_ANALYSIS.md` to reflect Phase 1 completion:

```markdown
## Phase 1 Completion Status

### ✅ **COMPLETED** - Modularize Core Agent

**What was accomplished:**
- Added 6 modular components to Core Agent
- Maintained 100% backward compatibility
- Enhanced internal architecture without breaking changes
- Added comprehensive testing

**New modular components:**
- `memory_manager` - Conversation memory management
- `streaming_manager` - Real-time streaming functionality
- `message_processor` - Message formatting and validation
- `response_processor` - Response processing and formatting
- `stats_manager` - Statistics and monitoring
- `trace_manager` - Tracing and debugging

**Benefits achieved:**
- Better code organization
- Improved maintainability
- Enhanced debugging capabilities
- Foundation for future improvements
```

## Success Criteria

### **Functional Requirements**
- ✅ Core Agent continues to work exactly as before
- ✅ All existing interfaces remain unchanged
- ✅ All existing functionality preserved
- ✅ No breaking changes to app.py integration

### **Technical Requirements**
- ✅ 6 modular components successfully integrated
- ✅ Memory management modularized
- ✅ Streaming functionality modularized
- ✅ Message processing modularized
- ✅ Response processing modularized
- ✅ Statistics tracking modularized
- ✅ Tracing functionality modularized

### **Quality Requirements**
- ✅ Comprehensive test coverage
- ✅ Documentation updated
- ✅ Code follows existing patterns
- ✅ No performance regression
- ✅ Error handling preserved

## Risk Mitigation

### **Low Risk Factors**
- No interface changes
- Incremental addition of components
- Existing functionality preserved
- Comprehensive testing

### **Mitigation Strategies**
- Test each component individually
- Test integration thoroughly
- Maintain backward compatibility
- Keep existing code as fallback

## Deliverables

### **Code Deliverables**
1. `agent_ng/memory_manager.py` - Memory management module
2. `agent_ng/streaming_manager.py` - Streaming functionality module
3. `agent_ng/message_processor.py` - Message processing module
4. `agent_ng/response_processor.py` - Response processing module
5. Enhanced `agent_ng/stats_manager.py` - Statistics module
6. Enhanced `agent_ng/trace_manager.py` - Tracing module
7. Updated `agent_ng/core_agent.py` - Core Agent with modular components

### **Test Deliverables**
1. `misc_files/test_phase_1_integration.py` - Integration tests
2. Updated existing tests to work with modular components

### **Documentation Deliverables**
1. Updated `docs/CORE_AGENT_VS_LANGCHAIN_AGENT_ANALYSIS.md`
2. This implementation plan document
3. Code comments and docstrings

## Next Steps (Phase 2)

After Phase 1 completion, proceed to Phase 2: **Enhance Memory Management**

**Phase 2 Objectives:**
- Replace `defaultdict(list)` with proper `ConversationMemoryManager`
- Add `ToolAwareMemory` for tool call context
- Keep existing interface, improve internal implementation
- Add LangChain-native memory patterns

**Phase 2 Duration:** Week 2 (5 days)
**Phase 2 Risk Level:** LOW - Internal improvements only

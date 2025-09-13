# Phase 1 Implementation Plan: LangChain Purity Foundation

## Objective
Replace fake streaming with true LangChain streaming and implement LangChain native patterns.

## Current State Analysis

### **Critical Issues to Fix**
1. **Fake Streaming**: `stream_message()` processes entire response first, then streams with artificial delays
2. **Non-LangChain Memory**: Uses custom `ConversationBufferMemory` instead of LangChain native
3. **Missing Callbacks**: No `BaseCallbackHandler` implementation for real-time events
4. **Simple Streaming Dependency**: Uses `simple_streaming.py` with artificial delays

## Implementation Steps

### **Step 1: Remove Simple Streaming Dependency**

#### **1.1 Identify Current Usage**
```python
# Current usage in langchain_agent.py:
from .simple_streaming import get_simple_streaming_manager

# In stream_message():
streaming_manager = get_simple_streaming_manager()
async for event in streaming_manager.stream_response_with_tools(response_text, ...):
```

#### **1.2 Replace with LangChain Native Streaming**
```python
# New implementation:
async def stream_message(self, message: str, conversation_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
    """True LangChain streaming with astream_events()"""
    
    if not self.llm_instance:
        yield {
            "type": "error",
            "content": "Agent not initialized",
            "metadata": {"error": "not_initialized"}
        }
        return
    
    try:
        # Get conversation chain
        chain = self._get_conversation_chain(conversation_id)
        
        # Format messages
        messages = self._format_messages(message, conversation_id)
        
        # Stream with LangChain native events
        async for event in chain.llm.astream_events(messages, version="v1"):
            event_type = event.get("event", "unknown")
            event_data = event.get("data", {})
            
            if event_type == "on_llm_stream":
                chunk = event_data.get("chunk", {})
                if hasattr(chunk, 'content') and chunk.content:
                    yield {
                        "type": "content",
                        "content": chunk.content,
                        "metadata": {"chunk_type": "llm_response"}
                    }
            
            elif event_type == "on_tool_start":
                tool_name = event_data.get("name", "unknown")
                yield {
                    "type": "tool_start",
                    "content": f"🔧 Using {tool_name}",
                    "metadata": {"tool_name": tool_name}
                }
            
            elif event_type == "on_tool_end":
                yield {
                    "type": "tool_end",
                    "content": "✅ Tool completed",
                    "metadata": {}
                }
            
            elif event_type == "on_llm_start":
                yield {
                    "type": "llm_start",
                    "content": "🤖 **LLM is thinking...**",
                    "metadata": {"llm_type": event_data.get("name", "unknown")}
                }
            
            elif event_type == "on_llm_end":
                yield {
                    "type": "llm_end",
                    "content": "✅ **LLM processing completed**",
                    "metadata": {}
                }
    
    except Exception as e:
        yield {
            "type": "error",
            "content": f"Error processing message: {str(e)}",
            "metadata": {"error": str(e)}
        }
```

### **Step 2: Implement LangChain Native Memory**

#### **2.1 Replace Custom Memory Implementation**
```python
# Remove custom ConversationBufferMemory
# class ConversationBufferMemory:  # ❌ REMOVE THIS

# Add LangChain native memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory.chat_message_histories import ChatMessageHistory

class LangChainMemoryManager:
    """LangChain native memory manager"""
    
    def __init__(self):
        self.memories = {}
    
    def get_memory(self, conversation_id: str):
        """Get or create memory for conversation"""
        if conversation_id not in self.memories:
            self.memories[conversation_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                chat_memory=ChatMessageHistory()
            )
        return self.memories[conversation_id]
    
    def clear_memory(self, conversation_id: str):
        """Clear memory for conversation"""
        if conversation_id in self.memories:
            self.memories[conversation_id].clear()
    
    def get_conversation_history(self, conversation_id: str):
        """Get conversation history"""
        memory = self.get_memory(conversation_id)
        return memory.load_memory_variables({})["chat_history"]
```

#### **2.2 Update Agent to Use Native Memory**
```python
class CmwAgent:
    def __init__(self, system_prompt: str = None):
        # ... existing initialization ...
        
        # Replace custom memory with LangChain native
        self.memory_manager = LangChainMemoryManager()
        
        # ... rest of initialization ...
    
    def _format_messages(self, message: str, conversation_id: str = "default") -> List[BaseMessage]:
        """Format messages using LangChain native memory"""
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Get conversation history from LangChain memory
        chat_history = self.memory_manager.get_conversation_history(conversation_id)
        messages.extend(chat_history)
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        return messages
    
    def _save_to_memory(self, conversation_id: str, user_message: str, ai_message: str):
        """Save conversation to LangChain memory"""
        memory = self.memory_manager.get_memory(conversation_id)
        memory.save_context(
            {"input": user_message},
            {"output": ai_message}
        )
```

### **Step 3: Add LangChain Callback Handlers**

#### **3.1 Create Streaming Callback Handler**
```python
class StreamingCallbackHandler(BaseCallbackHandler):
    """LangChain native callback handler for streaming"""
    
    def __init__(self, event_callback=None):
        self.event_callback = event_callback
        self.current_tool = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts"""
        if self.event_callback:
            self.event_callback({
                "type": "llm_start",
                "content": "🤖 **LLM is thinking...**",
                "metadata": {
                    "llm_type": serialized.get("name", "unknown"),
                    "prompts": prompts
                }
            })
    
    def on_llm_stream(self, chunk, **kwargs):
        """Called when LLM streams content"""
        if hasattr(chunk, 'content') and chunk.content and self.event_callback:
            self.event_callback({
                "type": "content",
                "content": chunk.content,
                "metadata": {
                    "chunk_type": "llm_response",
                    "tool_calls": getattr(chunk, 'tool_calls', [])
                }
            })
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts"""
        self.current_tool = serialized.get("name", "unknown_tool")
        if self.event_callback:
            self.event_callback({
                "type": "tool_start",
                "content": f"🔧 **Using tool: {self.current_tool}**",
                "metadata": {
                    "tool_name": self.current_tool,
                    "tool_args": input_str
                }
            })
    
    def on_tool_end(self, output, **kwargs):
        """Called when a tool ends"""
        if self.event_callback:
            self.event_callback({
                "type": "tool_end",
                "content": f"✅ **Tool completed: {self.current_tool}**",
                "metadata": {
                    "tool_name": self.current_tool,
                    "output": output
                }
            })
        self.current_tool = None
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends"""
        if self.event_callback:
            self.event_callback({
                "type": "llm_end",
                "content": "✅ **LLM processing completed**",
                "metadata": {
                    "response": response,
                    "tool_calls": getattr(response, 'tool_calls', [])
                }
            })
```

#### **3.2 Integrate Callback Handler**
```python
class CmwAgent:
    def __init__(self, system_prompt: str = None):
        # ... existing initialization ...
        
        # Add callback handler
        self.callback_handler = StreamingCallbackHandler(self._handle_streaming_event)
        
        # ... rest of initialization ...
    
    def _handle_streaming_event(self, event: Dict[str, Any]):
        """Handle streaming events from callbacks"""
        # This will be called by the callback handler
        # Store events for streaming or process them directly
        pass
    
    async def stream_message(self, message: str, conversation_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
        """Stream with LangChain callbacks"""
        
        if not self.llm_instance:
            yield {
                "type": "error",
                "content": "Agent not initialized",
                "metadata": {"error": "not_initialized"}
            }
            return
        
        try:
            # Get conversation chain
            chain = self._get_conversation_chain(conversation_id)
            
            # Format messages
            messages = self._format_messages(message, conversation_id)
            
            # Create event queue for streaming
            event_queue = asyncio.Queue()
            
            # Set up callback handler
            self.callback_handler.event_callback = lambda event: asyncio.create_task(event_queue.put(event))
            
            # Stream with callbacks
            async def stream_with_callbacks():
                async for event in chain.llm.astream_events(
                    messages,
                    version="v1",
                    callbacks=[self.callback_handler]
                ):
                    # Process LangChain events
                    event_type = event.get("event", "unknown")
                    event_data = event.get("data", {})
                    
                    if event_type == "on_llm_stream":
                        chunk = event_data.get("chunk", {})
                        if hasattr(chunk, 'content') and chunk.content:
                            yield {
                                "type": "content",
                                "content": chunk.content,
                                "metadata": {"chunk_type": "llm_response"}
                            }
                    
                    elif event_type == "on_tool_start":
                        tool_name = event_data.get("name", "unknown")
                        yield {
                            "type": "tool_start",
                            "content": f"🔧 Using {tool_name}",
                            "metadata": {"tool_name": tool_name}
                        }
                    
                    elif event_type == "on_tool_end":
                        yield {
                            "type": "tool_end",
                            "content": "✅ Tool completed",
                            "metadata": {}
                        }
            
            # Stream events
            async for event in stream_with_callbacks():
                yield event
            
            # Save to memory
            self._save_to_memory(conversation_id, message, "Response completed")
    
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "metadata": {"error": str(e)}
            }
```

### **Step 4: Update Conversation Chain**

#### **4.1 Create LangChain Native Chain**
```python
def create_langchain_conversation_chain(llm_instance, tools, system_prompt, memory_manager):
    """Create a pure LangChain conversation chain"""
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm_instance.llm.bind_tools(tools)
    
    return chain

class CmwAgent:
    def _get_conversation_chain(self, conversation_id: str = "default"):
        """Get or create conversation chain with LangChain native patterns"""
        if conversation_id not in self.conversation_chains:
            self.conversation_chains[conversation_id] = create_langchain_conversation_chain(
                self.llm_instance, 
                self.tools, 
                self.system_prompt,
                self.memory_manager
            )
        return self.conversation_chains[conversation_id]
```

### **Step 5: Remove Dependencies**

#### **5.1 Remove Simple Streaming Import**
```python
# Remove this line:
# from .simple_streaming import get_simple_streaming_manager

# Remove simple streaming usage in stream_message()
# Remove simple streaming usage in stream_chat()
```

#### **5.2 Update Imports**
```python
# Add new imports:
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

### **Step 6: Testing and Validation**

#### **6.1 Test Streaming**
```python
async def test_streaming():
    """Test the new streaming implementation"""
    agent = CmwAgent()
    
    # Test streaming
    async for event in agent.stream_message("Hello, how are you?"):
        print(f"Event: {event['type']} - {event['content']}")
        
        # Verify no artificial delays
        # Verify real-time streaming
        # Verify proper event types
```

#### **6.2 Test Memory**
```python
async def test_memory():
    """Test LangChain native memory"""
    agent = CmwAgent()
    
    # Test conversation
    await agent.stream_message("Hello, my name is John")
    await agent.stream_message("What's my name?")
    
    # Verify memory persistence
    history = agent.memory_manager.get_conversation_history("default")
    assert len(history) > 0
```

#### **6.3 Test Callbacks**
```python
async def test_callbacks():
    """Test callback handlers"""
    agent = CmwAgent()
    
    events = []
    agent.callback_handler.event_callback = lambda event: events.append(event)
    
    await agent.stream_message("Hello")
    
    # Verify callbacks were called
    assert len(events) > 0
    assert any(event["type"] == "llm_start" for event in events)
```

## Implementation Checklist

### **Phase 1 Tasks**
- [ ] **Remove simple_streaming.py dependency**
- [ ] **Implement LangChain native streaming with astream_events()**
- [ ] **Replace custom memory with LangChain native memory**
- [ ] **Add BaseCallbackHandler implementation**
- [ ] **Update conversation chain to use LangChain patterns**
- [ ] **Remove artificial delays**
- [ ] **Test streaming functionality**
- [ ] **Test memory persistence**
- [ ] **Test callback handlers**
- [ ] **Update documentation**

### **Success Criteria**
1. **No artificial delays** - streaming is truly real-time
2. **LangChain native patterns** - using official LangChain classes
3. **Real-time streaming** - content streams as it's generated
4. **Memory persistence** - conversations are properly stored
5. **Callback integration** - events are properly handled
6. **No simple_streaming dependency** - completely removed

### **Risk Mitigation**
1. **Incremental implementation** - test each step
2. **Fallback to current implementation** - if issues arise
3. **Comprehensive testing** - verify all functionality
4. **User feedback** - test with real users

## Expected Benefits

### **Immediate Benefits**
- **True real-time streaming** - no more fake streaming
- **LangChain native patterns** - better maintainability
- **Real-time events** - proper callback handling
- **Better memory management** - LangChain native features

### **Performance Benefits**
- **No artificial delays** - faster response times
- **Real-time streaming** - better user experience
- **LangChain optimization** - better resource utilization

### **Developer Benefits**
- **Standard patterns** - easier to maintain
- **LangChain ecosystem** - access to all features
- **Better debugging** - proper event handling
- **Future-proof** - follows LangChain best practices

## Conclusion

Phase 1 transforms the agent from **fake streaming** to **true LangChain streaming** by:

1. **Removing simple_streaming.py dependency**
2. **Implementing astream_events() for real-time streaming**
3. **Using LangChain native memory management**
4. **Adding proper callback handlers**
5. **Following LangChain best practices**

This provides the foundation for **Phase 2 (Parallel Tool Execution)** and **Phase 3 (Advanced Streaming & Thinking)**.

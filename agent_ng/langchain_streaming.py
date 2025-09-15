"""
LangChain Native Streaming Implementation
=======================================

A proper streaming implementation using LangChain's native streaming capabilities
with astream_events and proper event handling.
"""

import asyncio
from typing import Dict, Any, AsyncGenerator, List
from dataclasses import dataclass
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage


@dataclass
class LangChainStreamingEvent:
    """LangChain streaming event"""
    event_type: str
    content: str
    metadata: Dict[str, Any] = None


class LangChainStreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for LangChain streaming events"""
    
    def __init__(self):
        self.events = []
        self.current_content = ""
        self.tool_calls = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts"""
        self.events.append({
            "type": "llm_start",
            "content": "🤖 **Starting response generation...**",
            "metadata": {"prompts": prompts}
        })
    
    def on_llm_stream(self, chunk: BaseMessage, **kwargs) -> None:
        """Called when LLM streams content"""
        if hasattr(chunk, 'content') and chunk.content:
            self.current_content += chunk.content
            self.events.append({
                "type": "content",
                "content": chunk.content,
                "metadata": {"chunk": True}
            })
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts"""
        tool_name = serialized.get("name", "unknown")
        self.tool_calls.append(tool_name)
        
        # Filter out schema tools
        schema_tools = {'submit_answer', 'submit_intermediate_step'}
        if tool_name not in schema_tools:
            self.events.append({
                "type": "tool_start",
                "content": f"🔧 **Using tool: {tool_name}**",
                "metadata": {"tool_name": tool_name, "input": input_str}
            })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends"""
        if self.tool_calls:
            tool_name = self.tool_calls.pop()
            schema_tools = {'submit_answer', 'submit_intermediate_step'}
            if tool_name not in schema_tools:
                self.events.append({
                    "type": "tool_end",
                    "content": f"✅ **{tool_name} completed**",
                    "metadata": {"tool_name": tool_name, "output": output}
                })
    
    def on_llm_end(self, response: BaseMessage, **kwargs) -> None:
        """Called when LLM ends"""
        self.events.append({
            "type": "llm_end",
            "content": "✅ **Response generation completed**",
            "metadata": {"final_content": self.current_content}
        })
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain starts"""
        self.events.append({
            "type": "chain_start",
            "content": "🔄 **Processing request...**",
            "metadata": {"inputs": inputs}
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain ends"""
        self.events.append({
            "type": "chain_end",
            "content": "✅ **Processing completed**",
            "metadata": {"outputs": outputs}
        })
    
    def on_error(self, error: Exception, **kwargs) -> None:
        """Called when error occurs"""
        self.events.append({
            "type": "error",
            "content": f"❌ **Error: {str(error)}**",
            "metadata": {"error": str(error)}
        })


class LangChainStreamingManager:
    """Manager for LangChain native streaming"""
    
    def __init__(self):
        self.callback_handler = LangChainStreamingCallbackHandler()
    
    async def stream_agent_response(
        self, 
        agent, 
        message: str, 
        conversation_id: str = "default"
    ) -> AsyncGenerator[LangChainStreamingEvent, None]:
        """
        Stream agent response using LangChain native streaming
        
        Args:
            agent: The LangChain agent
            message: User message
            conversation_id: Conversation ID
            
        Yields:
            LangChainStreamingEvent objects
        """
        try:
            # Get the conversation chain
            chain = agent._get_conversation_chain(conversation_id)
            
            # Create the runnable chain
            runnable_chain = chain._create_chain()
            
            # Prepare the input with all required variables
            chat_history = agent.memory_manager.get_conversation_history(conversation_id)
            input_data = {
                "input": message,
                "chat_history": chat_history,
                "agent_scratchpad": []
            }
            
            # Stream using astream_events
            async for event in runnable_chain.astream_events(
                input_data,
                version="v1",
                include_names=["langchain"]
            ):
                event_type = event.get("event", "unknown")
                event_data = event.get("data", {})
                
                # Process different event types
                if event_type == "on_llm_start":
                    yield LangChainStreamingEvent(
                        event_type="thinking",
                        content="🤖 **Thinking...**",
                        metadata=event_data
                    )
                
                elif event_type == "on_llm_stream":
                    chunk = event_data.get("chunk", {})
                    if hasattr(chunk, 'content') and chunk.content:
                        yield LangChainStreamingEvent(
                            event_type="content",
                            content=chunk.content,
                            metadata={"chunk": True}
                        )
                
                elif event_type == "on_tool_start":
                    tool_name = event_data.get("name", "unknown")
                    schema_tools = {'submit_answer', 'submit_intermediate_step'}
                    if tool_name not in schema_tools:
                        yield LangChainStreamingEvent(
                            event_type="tool_start",
                            content=f"🔧 **Using tool: {tool_name}**",
                            metadata={"tool_name": tool_name}
                        )
                
                elif event_type == "on_tool_end":
                    tool_name = event_data.get("name", "unknown")
                    schema_tools = {'submit_answer', 'submit_intermediate_step'}
                    if tool_name not in schema_tools:
                        yield LangChainStreamingEvent(
                            event_type="tool_end",
                            content=f"✅ **{tool_name} completed**",
                            metadata={"tool_name": tool_name}
                        )
                
                elif event_type == "on_llm_end":
                    yield LangChainStreamingEvent(
                        event_type="completion",
                        content="✅ **Response completed**",
                        metadata=event_data
                    )
                
                elif event_type == "on_chain_error":
                    error = event_data.get("error", "Unknown error")
                    yield LangChainStreamingEvent(
                        event_type="error",
                        content=f"❌ **Error: {str(error)}**",
                        metadata={"error": str(error)}
                    )
                
                # Small delay for streaming effect
                await asyncio.sleep(0.01)
                
        except Exception as e:
            yield LangChainStreamingEvent(
                event_type="error",
                content=f"❌ **Streaming error: {str(e)}**",
                metadata={"error": str(e)}
            )


# Global streaming manager instance
_langchain_streaming_manager = None

def get_langchain_streaming_manager() -> LangChainStreamingManager:
    """Get the global LangChain streaming manager instance"""
    global _langchain_streaming_manager
    if _langchain_streaming_manager is None:
        _langchain_streaming_manager = LangChainStreamingManager()
    return _langchain_streaming_manager

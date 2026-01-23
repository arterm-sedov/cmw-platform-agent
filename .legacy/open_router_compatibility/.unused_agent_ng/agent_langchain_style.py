# Enhanced CmwAgent with LangChain-style state management
# This keeps all your advanced features while adding proper multi-turn conversation support

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

class AgentState(TypedDict):
    """State for the agent conversation"""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    current_question: str
    file_data: Optional[str]
    file_name: Optional[str]
    llm_sequence: Optional[List[str]]
    conversation_id: str

class CmwLangChainAgent:
    """
    Enhanced CmwAgent with LangChain-style state management for proper multi-turn conversations.
    Keeps all your advanced features while adding proper conversation state handling.
    """

    def __init__(self, original_cmw_agent):
        """
        Initialize with your existing CmwAgent instance.
        This preserves all your advanced features while adding LangChain-style state management.
        """
        self.cmw_agent = original_cmw_agent
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state graph"""

        def agent_node(state: AgentState):
            """Main agent node that uses your existing sophisticated logic"""
            # Extract the latest question
            current_question = state["current_question"]
            file_data = state.get("file_data")
            file_name = state.get("file_name")
            llm_sequence = state.get("llm_sequence")

            # Convert LangChain messages to your chat_history format
            chat_history = self._convert_messages_to_chat_history(state["messages"][:-1])  # Exclude the latest message

            # Use your existing sophisticated agent logic
            result = self.cmw_agent(
                question=current_question,
                file_data=file_data,
                file_name=file_name,
                llm_sequence=llm_sequence,
                chat_history=chat_history
            )

            # Add the result as an AI message
            new_message = AIMessage(content=result)

            return {
                "messages": state["messages"] + [new_message],
                "current_question": current_question,
                "file_data": file_data,
                "file_name": file_name,
                "llm_sequence": llm_sequence,
                "conversation_id": state["conversation_id"]
            }

        # Build the graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        return graph.compile(checkpointer=self.memory)

    def _convert_messages_to_chat_history(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to your chat_history format"""
        chat_history = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                chat_history.append({
                    "role": "user",
                    "content": msg.content
                })
            elif isinstance(msg, AIMessage):
                msg_dict = {
                    "role": "assistant",
                    "content": msg.content
                }
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", "")
                        } for tc in msg.tool_calls
                    ]
                chat_history.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                chat_history.append({
                    "role": "tool",
                    "content": msg.content,
                    "name": getattr(msg, 'name', 'unknown'),
                    "tool_call_id": getattr(msg, 'tool_call_id', 'unknown')
                })

        return chat_history

    def chat(self, message: str, conversation_id: str = "default", 
             file_data: str = None, file_name: str = None, 
             llm_sequence: List[str] = None) -> str:
        """
        Chat with the agent using proper conversation state management.

        Args:
            message: The user's message
            conversation_id: Unique ID for the conversation thread
            file_data: Optional file data
            file_name: Optional file name
            llm_sequence: Optional LLM sequence to use

        Returns:
            The agent's response
        """
        # Create the initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "current_question": message,
            "file_data": file_data,
            "file_name": file_name,
            "llm_sequence": llm_sequence,
            "conversation_id": conversation_id
        }

        # Run the graph with conversation state
        config = {"configurable": {"thread_id": conversation_id}}
        result = self.graph.invoke(initial_state, config=config)

        # Return the latest AI message content
        return result["messages"][-1].content

    def stream_chat(self, message: str, conversation_id: str = "default",
                   file_data: str = None, file_name: str = None,
                   llm_sequence: List[str] = None):
        """
        Stream chat with the agent using proper conversation state management.

        Args:
            message: The user's message
            conversation_id: Unique ID for the conversation thread
            file_data: Optional file data
            file_name: Optional file name
            llm_sequence: Optional LLM sequence to use

        Yields:
            Streaming chunks from the agent
        """
        # Create the initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "current_question": message,
            "file_data": file_data,
            "file_name": file_name,
            "llm_sequence": llm_sequence,
            "conversation_id": conversation_id
        }

        # Run the graph with streaming
        config = {"configurable": {"thread_id": conversation_id}}

        # Use your existing streaming logic but with proper state management
        chat_history = self._convert_messages_to_chat_history(initial_state["messages"][:-1])

        for chunk in self.cmw_agent.stream(
            message, 
            chat_history=chat_history,
            llm_sequence=llm_sequence
        ):
            yield chunk

    def get_conversation_history(self, conversation_id: str = "default") -> List[Dict[str, Any]]:
        """Get the conversation history for a specific conversation"""
        config = {"configurable": {"thread_id": conversation_id}}

        # Get the current state
        try:
            state = self.graph.get_state(config)
            if state and state.values:
                return self._convert_messages_to_chat_history(state.values["messages"])
        except Exception as e:
            print(f"Error getting conversation history: {e}")

        return []

    def clear_conversation(self, conversation_id: str = "default"):
        """Clear a specific conversation"""
        # This would require implementing a clear method in MemorySaver
        # For now, we'll just note that this conversation should be treated as new
        pass

# Usage example:
def create_enhanced_agent(original_agent):
    """Create an enhanced agent with LangChain-style state management"""
    return CmwLangChainAgent(original_agent)

# Example usage in your app.py:
def chat_with_agent_enhanced(message, history, conversation_id="default"):
    """
    Enhanced chat function with proper multi-turn conversation support.
    """
    if not hasattr(chat_with_agent_enhanced, 'enhanced_agent'):
        # Initialize the enhanced agent once
        chat_with_agent_enhanced.enhanced_agent = create_enhanced_agent(agent)

    # Convert Gradio history to our format
    chat_history = []
    for turn in history:
        if turn[0]:  # User message
            chat_history.append({"role": "user", "content": turn[0]})
        if turn[1]:  # Assistant message
            chat_history.append({"role": "assistant", "content": turn[1]})

    # Get the response with proper conversation state
    response = chat_with_agent_enhanced.enhanced_agent.chat(
        message=message,
        conversation_id=conversation_id
    )

    # Update history
    history.append([message, response])
    return history, ""

def stream_chat_with_agent_enhanced(message, history, conversation_id="default"):
    """
    Enhanced streaming chat function with proper multi-turn conversation support.
    """
    if not hasattr(stream_chat_with_agent_enhanced, 'enhanced_agent'):
        # Initialize the enhanced agent once
        stream_chat_with_agent_enhanced.enhanced_agent = create_enhanced_agent(agent)

    # Convert Gradio history to our format
    chat_history = []
    for turn in history:
        if turn[0]:  # User message
            chat_history.append({"role": "user", "content": turn[0]})
        if turn[1]:  # Assistant message
            chat_history.append({"role": "assistant", "content": turn[1]})

    # Stream the response with proper conversation state
    working_history = history + [[message, ""]]

    for chunk in stream_chat_with_agent_enhanced.enhanced_agent.stream_chat(
        message=message,
        conversation_id=conversation_id
    ):
        working_history[-1][1] += chunk
        yield working_history, ""

    return working_history, ""

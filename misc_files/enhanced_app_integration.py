# Enhanced app.py integration for multi-turn conversations
# This shows how to fix the multi-turn conversation issue while keeping your advanced features

import gradio as gr
from agent_langchain_style import CmwLangChainAgent
import threading

# Global variables
agent = None
enhanced_agent = None
agent_lock = threading.Lock()

def initialize_enhanced_agent(original_agent):
    """Initialize the enhanced agent with LangChain-style state management"""
    global enhanced_agent
    if enhanced_agent is None:
        enhanced_agent = CmwLangChainAgent(original_agent)
    return enhanced_agent

def chat_with_agent_enhanced(message, history, conversation_id="default"):
    """
    Enhanced chat function with proper multi-turn conversation support.
    This fixes the multi-turn conversation issue while keeping all your advanced features.
    """
    global enhanced_agent
    
    if not message.strip():
        return history, ""
    
    if agent is None:
        return history + [[message, "Error: Agent not initialized. Check logs for details."]], ""
    
    # Initialize enhanced agent if needed
    if enhanced_agent is None:
        enhanced_agent = initialize_enhanced_agent(agent)
    
    # Acquire lock to prevent concurrent agent calls
    if not agent_lock.acquire(blocking=False):
        return history + [[message, "⚠️ Another request is being processed. Please wait..."]], ""
    
    try:
        print(f"💬 Enhanced Chat request: {message}")
        
        # Get the response with proper conversation state management
        response = enhanced_agent.chat(
            message=message,
            conversation_id=conversation_id
        )
        
        # Update history
        history.append([message, response])
        return history, ""
        
    except Exception as e:
        print(f"Error in enhanced chat: {e}")
        return history + [[message, f"Error: {str(e)}"]], ""
    finally:
        agent_lock.release()

def stream_chat_with_agent_enhanced(message, history, conversation_id="default"):
    """
    Enhanced streaming chat function with proper multi-turn conversation support.
    This provides real-time streaming while maintaining conversation context.
    """
    global enhanced_agent
    
    if not message.strip():
        yield history, ""
        return
    
    if agent is None:
        yield history + [[message, "Error: Agent not initialized. Check logs for details."]], ""
        return
    
    # Initialize enhanced agent if needed
    if enhanced_agent is None:
        enhanced_agent = initialize_enhanced_agent(agent)
    
    # Acquire lock to prevent concurrent agent calls
    if not agent_lock.acquire(blocking=False):
        yield history + [[message, "⚠️ Another request is being processed. Please wait..."]], ""
        return
    
    try:
        print(f"💬 Enhanced Stream Chat request: {message}")
        
        # Start with the user message
        working_history = history + [[message, ""]]
        yield working_history, ""
        
        # Stream the response with proper conversation state management
        accumulated_response = ""
        for chunk in enhanced_agent.stream_chat(
            message=message,
            conversation_id=conversation_id
        ):
            accumulated_response += chunk
            working_history[-1][1] = accumulated_response
            yield working_history, ""
        
        # Final yield
        yield working_history, ""
        
    except Exception as e:
        print(f"Error in enhanced stream chat: {e}")
        yield history + [[message, f"Error: {str(e)}"]], ""
    finally:
        agent_lock.release()

def get_conversation_history(conversation_id="default"):
    """Get the conversation history for debugging"""
    global enhanced_agent
    if enhanced_agent is None:
        return []
    return enhanced_agent.get_conversation_history(conversation_id)

def clear_conversation(conversation_id="default"):
    """Clear a conversation"""
    global enhanced_agent
    if enhanced_agent is not None:
        enhanced_agent.clear_conversation(conversation_id)

# Example Gradio interface with conversation ID support
def create_enhanced_interface():
    """Create a Gradio interface with proper multi-turn conversation support"""
    
    with gr.Blocks(title="CMW Platform Agent - Enhanced") as interface:
        gr.Markdown("# CMW Platform Agent - Enhanced Multi-Turn Conversations")
        
        # Conversation ID input
        conversation_id = gr.Textbox(
            label="Conversation ID", 
            value="default", 
            placeholder="Enter a unique conversation ID"
        )
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Chat History",
            height=600,
            show_label=True
        )
        
        msg = gr.Textbox(
            label="Message",
            placeholder="Ask me anything about the CMW platform...",
            lines=2
        )
        
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Conversation")
            stream_btn = gr.Button("Stream Response", variant="secondary")
        
        # Debug info
        with gr.Accordion("Debug Info", open=False):
            debug_output = gr.Textbox(
                label="Conversation History (JSON)",
                lines=10,
                interactive=False
            )
            refresh_debug_btn = gr.Button("Refresh Debug Info")
        
        # Event handlers
        def send_message(message, history, conv_id):
            return chat_with_agent_enhanced(message, history, conv_id)
        
        def stream_message(message, history, conv_id):
            return stream_chat_with_agent_enhanced(message, history, conv_id)
        
        def clear_chat(conv_id):
            clear_conversation(conv_id)
            return [], ""
        
        def refresh_debug(conv_id):
            history = get_conversation_history(conv_id)
            return json.dumps(history, indent=2, ensure_ascii=False)
        
        # Connect events
        send_btn.click(
            send_message,
            inputs=[msg, chatbot, conversation_id],
            outputs=[chatbot, msg]
        )
        
        stream_btn.click(
            stream_message,
            inputs=[msg, chatbot, conversation_id],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_chat,
            inputs=[conversation_id],
            outputs=[chatbot, msg]
        )
        
        refresh_debug_btn.click(
            refresh_debug,
            inputs=[conversation_id],
            outputs=[debug_output]
        )
        
        # Enter key support
        msg.submit(
            send_message,
            inputs=[msg, chatbot, conversation_id],
            outputs=[chatbot, msg]
        )
    
    return interface

# Example of how to integrate with your existing app.py
def integrate_with_existing_app():
    """
    Example of how to integrate this with your existing app.py
    """
    # In your existing app.py, replace the chat functions with these:
    
    # Replace chat_with_agent_stream with:
    # chat_with_agent_stream = stream_chat_with_agent_enhanced
    
    # Replace any other chat functions with:
    # chat_with_agent = chat_with_agent_enhanced
    
    # Add conversation ID support to your Gradio interface
    pass

# Global interface variable for Gradio reload functionality
# Initialize interface immediately to avoid NoneType errors during reload
interface = create_enhanced_interface()

if __name__ == "__main__":
    # Example usage
    import json
    
    # This would be your existing agent initialization
    # agent = CmwAgent(provider="openrouter")
    
    # Launch the enhanced interface
    interface.launch()

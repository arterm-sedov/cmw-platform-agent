#!/usr/bin/env python3
"""
Gradio integration example using the new LangChain-compatible methods
This shows how to fix multi-turn conversations in your existing app.py
"""

import gradio as gr
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def create_langchain_gradio_interface():
    """
    Create a Gradio interface that uses the new LangChain-compatible methods
    This fixes the multi-turn conversation issue
    """
    
    # Import your agent (adjust the import as needed)
    try:
        from agent import CmwAgent
        print("✅ Successfully imported CmwAgent")
    except ImportError as e:
        print(f"❌ Failed to import CmwAgent: {e}")
        return None
    
    # Initialize agent
    try:
        agent = CmwAgent(provider="openrouter")  # Adjust provider as needed
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return None
    
    def chat_with_agent_langchain(message, history):
        """
        Enhanced chat function using LangChain-compatible invoke method.
        This properly handles multi-turn conversations.
        """
        if not message.strip():
            return history, ""
        
        try:
            # Convert Gradio history format to our chat_history format
            chat_history = []
            for turn in history:
                if turn[0]:  # User message
                    chat_history.append({"role": "user", "content": turn[0]})
                if turn[1]:  # Assistant message
                    chat_history.append({"role": "assistant", "content": turn[1]})
            
            # Use the LangChain-compatible invoke method
            result = agent.invoke({
                "input": message,
                "chat_history": chat_history
            })
            
            # Extract the response
            response = result.get("output", "No response generated")
            
            # Update history
            history.append([message, response])
            return history, ""
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append([message, error_msg])
            return history, ""
    
    def stream_chat_with_agent_langchain(message, history):
        """
        Enhanced streaming chat function using LangChain-compatible astream method.
        This provides real-time streaming with proper multi-turn conversation support.
        """
        if not message.strip():
            yield history, ""
            return
        
        try:
            # Convert Gradio history format to our chat_history format
            chat_history = []
            for turn in history:
                if turn[0]:  # User message
                    chat_history.append({"role": "user", "content": turn[0]})
                if turn[1]:  # Assistant message
                    chat_history.append({"role": "assistant", "content": turn[1]})
            
            # Start with the user message
            working_history = history + [[message, ""]]
            yield working_history, ""
            
            # Use the LangChain-compatible astream method
            accumulated_response = ""
            for chunk in agent.astream({
                "input": message,
                "chat_history": chat_history
            }):
                chunk_text = chunk.get("chunk", "")
                accumulated_response += chunk_text
                working_history[-1][1] = accumulated_response
                yield working_history, ""
            
            # Final yield
            yield working_history, ""
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield history + [[message, error_msg]], ""
    
    def clear_conversation():
        """Clear the conversation history"""
        return [], ""
    
    # Create the Gradio interface
    with gr.Blocks(title="CMW Platform Agent - LangChain Enhanced") as interface:
        gr.Markdown("# CMW Platform Agent - Enhanced Multi-Turn Conversations")
        gr.Markdown("This version uses LangChain-compatible methods for proper conversation state management.")
        
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
            stream_btn = gr.Button("Stream Response", variant="secondary")
            clear_btn = gr.Button("Clear Conversation")
        
        # Debug info
        with gr.Accordion("Debug Info", open=False):
            debug_output = gr.Textbox(
                label="Agent Status",
                value="Agent initialized successfully",
                lines=5,
                interactive=False
            )
        
        # Event handlers
        send_btn.click(
            chat_with_agent_langchain,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        stream_btn.click(
            stream_chat_with_agent_langchain,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, msg]
        )
        
        # Enter key support
        msg.submit(
            chat_with_agent_langchain,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
    
    return interface

def show_integration_instructions():
    """Show instructions for integrating with existing app.py"""
    print("\n🔧 Integration Instructions for app.py")
    print("=" * 50)
    print()
    print("To integrate these methods into your existing app.py:")
    print()
    print("1. Replace your chat_with_agent_stream function with:")
    print("   def chat_with_agent_stream(message, history):")
    print("       return chat_with_agent_langchain(message, history)")
    print()
    print("2. Add the chat_with_agent_langchain function from this file")
    print()
    print("3. For streaming, use stream_chat_with_agent_langchain")
    print()
    print("4. This will provide proper multi-turn conversation support!")
    print()
    print("Key benefits:")
    print("✅ Proper conversation state management")
    print("✅ LangChain compatibility")
    print("✅ Keeps all your advanced features")
    print("✅ Easy integration with existing code")

# Global interface variable for Gradio reload functionality
# Initialize interface immediately to avoid NoneType errors during reload
interface = create_langchain_gradio_interface()

if __name__ == "__main__":
    print("🚀 Creating LangChain-Enhanced Gradio Interface")
    print("=" * 50)
    
    if interface:
        print("\n✅ Interface created successfully!")
        print("Starting Gradio interface...")
        interface.launch()
    else:
        print("\n❌ Failed to create interface")
        show_integration_instructions()

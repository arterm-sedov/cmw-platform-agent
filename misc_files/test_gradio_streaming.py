"""
Gradio Streaming Test
====================

Test the streaming functionality in a Gradio context.
"""

import asyncio
import sys
import os
import gradio as gr

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.simple_streaming import get_simple_streaming_manager


async def test_streaming_generator():
    """Test streaming generator for Gradio"""
    streaming_manager = get_simple_streaming_manager()
    
    # Simulate streaming response
    response_text = "Hello! I'll help you calculate 15 * 23 + 7 step by step.\n\nFirst, let me multiply 15 by 23:\n15 * 23 = 345\n\nThen, let me add 7:\n345 + 7 = 352\n\nSo the answer is 352."
    
    # Stream the response
    async for event in streaming_manager.stream_text(response_text):
        yield event.content


def create_gradio_interface():
    """Create a simple Gradio interface to test streaming"""
    
    async def stream_chat(message, history):
        """Stream chat function"""
        if not message.strip():
            yield history, ""
            return
        
        # Add user message to history
        working_history = history + [(message, "")]
        yield working_history, ""
        
        # Stream response
        response_content = ""
        async for chunk in test_streaming_generator():
            response_content += chunk
            working_history[-1] = (message, response_content)
            yield working_history, ""
    
    # Create interface
    with gr.Blocks() as demo:
        gr.Markdown("# Streaming Test")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Message")
        clear = gr.Button("Clear")
        
        msg.submit(stream_chat, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(debug=True, share=False)

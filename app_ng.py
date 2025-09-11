"""
Next-Generation Gradio App
=========================

A clean, modern Gradio application that uses the latest features and best practices
for LLM agent interactions with real-time streaming and tool usage visualization.

Key Features:
- Modern Gradio ChatInterface and Blocks
- Real-time streaming with metadata
- Clean initialization and error handling
- Tool usage visualization
- Copy to clipboard functionality
- Responsive design with custom CSS
- Async LLM initialization
- Conversation management

Based on:
- https://www.gradio.app/guides/agents-and-tool-usage
- https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
- https://www.gradio.app/guides/chatinterface-examples
"""

import asyncio
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from dataclasses import asdict

# Local imports
from agent_ng import NextGenAgent, ChatMessage, get_agent_ng
from llm_manager import get_llm_manager


class NextGenApp:
    """Next-generation Gradio application with modern features"""
    
    def __init__(self):
        self.agent: Optional[NextGenAgent] = None
        self.llm_manager = get_llm_manager()
        self.initialization_logs = []
        self.is_initializing = False
        self.initialization_complete = False
        
        # Initialize synchronously first, then start async initialization
        self._start_async_initialization()
    
    def _start_async_initialization(self):
        """Start async initialization in a new event loop"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, schedule the task
            loop.create_task(self._initialize_agent())
        except RuntimeError:
            # No event loop running, start a new one
            import threading
            def run_async_init():
                asyncio.run(self._initialize_agent())
            
            thread = threading.Thread(target=run_async_init, daemon=True)
            thread.start()
    
    async def _initialize_agent(self):
        """Initialize the agent asynchronously"""
        self.is_initializing = True
        self.initialization_logs.append("🚀 Starting agent initialization...")
        
        try:
            # Initialize agent (uses single provider from AGENT_PROVIDER)
            self.agent = await get_agent_ng()
            
            # Wait for agent to be ready
            max_wait = 30  # 30 seconds timeout
            wait_time = 0
            while not self.agent.is_ready() and wait_time < max_wait:
                await asyncio.sleep(0.5)
                wait_time += 0.5
                self.initialization_logs.append(f"⏳ Waiting for agent... ({wait_time:.1f}s)")
            
            if self.agent.is_ready():
                status = self.agent.get_status()
                self.initialization_logs.append(f"✅ Agent ready with {status['current_llm']}")
                self.initialization_logs.append(f"🔧 Tools available: {status['tools_count']}")
                self.initialization_complete = True
            else:
                self.initialization_logs.append("❌ Agent initialization timeout")
                
        except Exception as e:
            self.initialization_logs.append(f"❌ Initialization failed: {str(e)}")
        
        self.is_initializing = False
    
    def get_initialization_logs(self) -> str:
        """Get initialization logs as formatted string"""
        return "\n".join(self.initialization_logs)
    
    def get_agent_status(self) -> str:
        """Get current agent status"""
        if not self.agent:
            return "🟡 Initializing agent..."
        
        status = self.agent.get_status()
        if status["is_ready"]:
            llm_info = self.agent.get_llm_info()
            return f"✅ **{llm_info['provider']}** ({llm_info['model_name']})\n🔧 {status['tools_count']} tools available"
        else:
            return "❌ Agent not ready"
    
    async def chat_with_agent(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Chat with the agent using modern streaming.
        
        Args:
            message: User message
            history: Chat history as list of tuples
            
        Returns:
            Updated history and empty message
        """
        if not self.agent or not self.agent.is_ready():
            error_msg = "Agent not ready. Please wait for initialization to complete."
            history.append((message, error_msg))
            return history, ""
        
        # Convert history to ChatMessage format
        chat_history = []
        for user_msg, assistant_msg in history:
            chat_history.append(ChatMessage(role="user", content=user_msg))
            chat_history.append(ChatMessage(role="assistant", content=assistant_msg))
        
        # Add current user message
        chat_history.append(ChatMessage(role="user", content=message))
        
        # Stream response
        response_content = ""
        thinking_content = ""
        tool_usage = []
        
        try:
            async for event in self.agent.stream_chat(message, chat_history[:-1]):  # Exclude current message
                if event["type"] == "content":
                    response_content += event["content"]
                elif event["type"] == "thinking":
                    thinking_content = event["content"]
                elif event["type"] == "tool_use":
                    tool_usage.append(event["content"])
                elif event["type"] == "error":
                    response_content = f"❌ {event['content']}"
                    break
        
        except Exception as e:
            response_content = f"❌ Error: {str(e)}"
        
        # Add response to history
        history.append((message, response_content))
        
        return history, ""
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .hero-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .status-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        
        .tool-usage {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #28a745;
            font-size: 0.9rem;
        }
        
        .thinking-content {
            background: #fff3cd;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #ffc107;
            font-style: italic;
        }
        
        .cmw-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .cmw-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .sidebar-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .chat-hints {
            background: #e3f2fd;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #2196f3;
        }
        """
        
        with gr.Blocks(
            css=custom_css,
            title="Next-Gen LLM Agent",
            theme=gr.themes.Soft()
        ) as demo:
            
            # Header
            gr.Markdown("# Next-Gen LLM Agent", elem_classes=["hero-title"])
            
            with gr.Tabs():
                
                # Chat Tab
                with gr.TabItem("💬 Chat", id="chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # Chat interface
                            chatbot = gr.Chatbot(
                                label="Chat with the Agent",
                                height=500,
                                show_label=True,
                                container=True,
                                show_copy_button=True
                            )
                            
                            with gr.Row():
                                msg = gr.Textbox(
                                    label="Your Message",
                                    placeholder="Type your message here...",
                                    lines=2,
                                    scale=4,
                                    max_lines=4
                                )
                                send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["cmw-button"])
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat", variant="secondary", elem_classes=["cmw-button"])
                                copy_btn = gr.Button("Copy Last Response", variant="secondary", elem_classes=["cmw-button"])
                        
                        with gr.Column(scale=1, elem_classes=["sidebar-card"]):
                            # Agent status
                            gr.Markdown("### 🤖 Agent Status")
                            status_display = gr.Markdown("🟡 Initializing...", elem_classes=["status-card"])
                            
                            # Quick actions
                            gr.Markdown("### ⚡ Quick Actions")
                            with gr.Column():
                                quick_math_btn = gr.Button("🧮 Math Question", elem_classes=["cmw-button"])
                                quick_code_btn = gr.Button("💻 Code Question", elem_classes=["cmw-button"])
                                quick_general_btn = gr.Button("💭 General Question", elem_classes=["cmw-button"])
                            
                            # Model info
                            gr.Markdown("### 📊 Model Info")
                            model_info = gr.Markdown("Loading...", elem_classes=["status-card"])
                    
                    # Event handlers
                    def clear_chat():
                        return [], ""
                    
                    def copy_last_response(history):
                        if history and len(history) > 0:
                            last_response = history[-1][1]  # Get last assistant message
                            return last_response
                        return ""
                    
                    def quick_math():
                        return "What is 15 * 23 + 7? Please show your work step by step."
                    
                    def quick_code():
                        return "Write a Python function to check if a number is prime. Include tests."
                    
                    def quick_general():
                        return "Explain the concept of machine learning in simple terms."
                    
                    # Connect event handlers
                    send_btn.click(
                        fn=self.chat_with_agent,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    msg.submit(
                        fn=self.chat_with_agent,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    clear_btn.click(
                        fn=clear_chat,
                        outputs=[chatbot, msg]
                    )
                    
                    copy_btn.click(
                        fn=copy_last_response,
                        inputs=[chatbot],
                        outputs=[msg]
                    )
                    
                    quick_math_btn.click(
                        fn=quick_math,
                        outputs=[msg]
                    )
                    
                    quick_code_btn.click(
                        fn=quick_code,
                        outputs=[msg]
                    )
                    
                    quick_general_btn.click(
                        fn=quick_general,
                        outputs=[msg]
                    )
                
                # Logs Tab
                with gr.TabItem("📜 Initialization Logs", id="logs"):
                    gr.Markdown("### Agent Initialization Logs")
                    logs_display = gr.Markdown(
                        "🟡 Starting initialization...",
                        elem_classes=["status-card"]
                    )
                    
                    refresh_logs_btn = gr.Button("🔄 Refresh Logs", elem_classes=["cmw-button"])
                    
                    def refresh_logs():
                        return self.get_initialization_logs()
                    
                    refresh_logs_btn.click(
                        fn=refresh_logs,
                        outputs=[logs_display]
                    )
                
                # Stats Tab
                with gr.TabItem("📊 Statistics", id="stats"):
                    gr.Markdown("### Agent Statistics")
                    stats_display = gr.Markdown("Loading statistics...", elem_classes=["status-card"])
                    
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", elem_classes=["cmw-button"])
                    
                    def refresh_stats():
                        if self.agent:
                            stats = self.agent.get_stats()
                            return f"""
                            **Agent Status:**
                            - Ready: {stats['agent_status']['is_ready']}
                            - LLM: {stats['llm_info'].get('model_name', 'Unknown')}
                            - Provider: {stats['llm_info'].get('provider', 'Unknown')}
                            
                            **Conversation:**
                            - Messages: {stats['conversation_stats']['message_count']}
                            - User: {stats['conversation_stats']['user_messages']}
                            - Assistant: {stats['conversation_stats']['assistant_messages']}
                            
                            **Tools:**
                            - Available: {stats['agent_status']['tools_count']}
                            """
                        return "Agent not available"
                    
                    refresh_stats_btn.click(
                        fn=refresh_stats,
                        outputs=[stats_display]
                    )
            
            # Auto-refresh status
            def update_status():
                return self.get_agent_status()
            
            def update_model_info():
                if self.agent:
                    llm_info = self.agent.get_llm_info()
                    return f"""
                    **Provider:** {llm_info.get('provider', 'Unknown')}
                    **Model:** {llm_info.get('model_name', 'Unknown')}
                    **Status:** {'✅ Healthy' if llm_info.get('is_healthy', False) else '❌ Unhealthy'}
                    **Last Used:** {time.ctime(llm_info.get('last_used', 0))}
                    """
                return "Agent not available"
            
            # Timer for auto-refresh
            status_timer = gr.Timer(2.0, active=True)
            status_timer.tick(
                fn=update_status,
                outputs=[status_display]
            )
            
            status_timer.tick(
                fn=update_model_info,
                outputs=[model_info]
            )
            
            # Load initial logs
            demo.load(
                fn=refresh_logs,
                outputs=[logs_display]
            )
        
        return demo


def main():
    """Main function to run the application"""
    print("🚀 Starting Next-Gen LLM Agent App...")
    
    # Create the app
    app = NextGenApp()
    
    # Create and launch the interface
    demo = app.create_interface()
    
    print("🌐 Launching Gradio interface...")
    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()

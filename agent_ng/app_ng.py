"""
LangChain-Native Gradio App
===========================

A modern Gradio application using pure LangChain patterns for multi-turn conversations
with tool calls, memory management, and streaming.

Key Features:
- Pure LangChain conversation chains and memory
- Multi-turn conversation support with tool calls
- Real-time streaming with metadata
- Native LangChain tool calling
- Modern Gradio UI with comprehensive monitoring
- Tool usage visualization
- Debug logging and statistics
- Responsive design with custom CSS

Based on LangChain's official documentation and best practices.
"""

import asyncio
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import json
import time
from dataclasses import asdict

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Local imports with robust fallback handling
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try absolute imports first (works from root directory)
try:
    from agent_ng.langchain_agent import CmwAgent as NextGenAgent, ChatMessage, get_agent_ng
    from agent_ng.llm_manager import get_llm_manager
    from agent_ng.debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
    from agent_ng.streaming_chat import get_chat_interface
except ImportError:
    # Fallback to relative imports (when running as module)
    try:
        from .langchain_agent import CmwAgent as NextGenAgent, ChatMessage, get_agent_ng
        from .llm_manager import get_llm_manager
        from .debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
        from .streaming_chat import get_chat_interface
    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")
        # Set defaults to prevent further errors
        NextGenAgent = None
        ChatMessage = None
        get_agent_ng = lambda: None
        get_llm_manager = lambda: None
        get_debug_streamer = lambda x: None
        get_log_handler = lambda: None
        LogLevel = None
        LogCategory = None
        get_chat_interface = lambda: None


class NextGenApp:
    """LangChain-native Gradio application with modern features"""
    
    def __init__(self):
        self.agent: Optional[NextGenAgent] = None
        self.llm_manager = get_llm_manager()
        self.initialization_logs = []
        self.is_initializing = False
        self.initialization_complete = False
        self.session_id = "default"  # LangChain session management
        
        # Initialize debug system
        self.debug_streamer = get_debug_streamer("app_ng")
        self.log_handler = get_log_handler("app_ng")
        self.chat_interface = get_chat_interface("app_ng")
        
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
        self.debug_streamer.info("Starting agent initialization", LogCategory.INIT)
        self.initialization_logs.append("🚀 Starting agent initialization...")
        
        try:
            # Initialize agent (uses single provider from AGENT_PROVIDER)
            self.debug_streamer.info("Creating agent instance", LogCategory.INIT)
            self.agent = await get_agent_ng()
            
            # Wait for agent to be ready
            max_wait = 30  # 30 seconds timeout
            wait_time = 0
            while not self.agent.is_ready() and wait_time < max_wait:
                await asyncio.sleep(0.5)
                wait_time += 0.5
                self.debug_streamer.debug(f"Waiting for agent... ({wait_time:.1f}s)", LogCategory.INIT)
                self.initialization_logs.append(f"⏳ Waiting for agent... ({wait_time:.1f}s)")
            
            if self.agent.is_ready():
                status = self.agent.get_status()
                self.debug_streamer.success(f"Agent ready with {status['current_llm']}", LogCategory.INIT)
                self.initialization_logs.append(f"✅ Agent ready with {status['current_llm']}")
                self.initialization_logs.append(f"🔧 Tools available: {status['tools_count']}")
                self.initialization_complete = True
            else:
                self.debug_streamer.error("Agent initialization timeout", LogCategory.INIT)
                self.initialization_logs.append("❌ Agent initialization timeout")
                
        except Exception as e:
            self.debug_streamer.error(f"Initialization failed: {str(e)}", LogCategory.INIT)
            self.initialization_logs.append(f"❌ Initialization failed: {str(e)}")
        
        self.is_initializing = False
    
    def get_initialization_logs(self) -> str:
        """Get initialization logs as formatted string"""
        # Combine static logs with real-time debug logs
        static_logs = "\n".join(self.initialization_logs)
        debug_logs = self.log_handler.get_current_logs()
        
        if debug_logs and debug_logs != "No logs available yet.":
            return f"{static_logs}\n\n--- Real-time Debug Logs ---\n\n{debug_logs}"
        return static_logs
    
    def get_agent_status(self) -> str:
        """Get current agent status with comprehensive details"""
        if not self.agent:
            return "🟡 Initializing agent..."
        
        status = self.agent.get_status()
        if status["is_ready"]:
            llm_info = self.agent.get_llm_info()
            return f"""✅ **Agent Ready**

**Provider:** {llm_info.get('provider', 'Unknown')}
**Model:** {llm_info.get('model_name', 'Unknown')}
**Status:** {'✅ Healthy' if llm_info.get('is_healthy', False) else '❌ Unhealthy'}
**Tools:** {status['tools_count']} available
**Last Used:** {time.ctime(llm_info.get('last_used', 0))}"""
        else:
            return "❌ Agent not ready"
    
    def is_ready(self) -> bool:
        """Check if the app is ready (LangChain-native pattern)"""
        return self.initialization_complete and self.agent is not None and self.agent.is_ready()
    
    def clear_conversation(self) -> Tuple[List[Dict[str, str]], str]:
        """Clear the conversation history (LangChain-native pattern)"""
        if self.agent:
            self.agent.clear_conversation(self.session_id)
            self.debug_streamer.info("Conversation cleared")
        return [], ""
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history (LangChain-native pattern)"""
        if self.agent:
            return self.agent.get_conversation_history(self.session_id)
        return []
    
    async def chat_with_agent(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        """
        Chat with the agent using LangChain-native patterns.
        
        Args:
            message: User message
            history: Chat history as list of message dicts
            
        Returns:
            Updated history and empty message
        """
        if not self.is_ready():
            error_msg = "❌ **Agent not ready. Please wait for initialization to complete.**"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
        
        if not message.strip():
            return history, ""
        
        try:
            self.debug_streamer.info(f"Processing message: {message[:50]}...")
            
            # Process message with LangChain agent
            response = self.agent.process_message(message, self.session_id)
            
            if response.success:
                # Add to history
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response.answer})
                
                # Log tool calls if any
                if response.tool_calls:
                    self.debug_streamer.info(f"Tool calls made: {[call['name'] for call in response.tool_calls]}")
                
                return history, ""
            else:
                error_msg = f"❌ **Error: {response.error}**"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                return history, ""
                
        except Exception as e:
            self.debug_streamer.error(f"Error in chat: {e}")
            error_msg = f"❌ **Error processing message: {str(e)}**"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    async def stream_chat_with_agent(self, message: str, history: List[Dict[str, str]]) -> AsyncGenerator[Tuple[List[Dict[str, str]], str], None]:
        """
        Stream chat with the agent using LangChain-native streaming patterns.
        
        Args:
            message: User message
            history: Chat history
            
        Yields:
            Updated history and empty message
        """
        if not self.is_ready():
            error_msg = "❌ **Agent not ready. Please wait for initialization to complete.**"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            yield history, ""
            return
        
        if not message.strip():
            yield history, ""
            return
        
        try:
            self.debug_streamer.info(f"Streaming message: {message[:50]}...")
            
            # Add user message to history
            working_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            yield working_history, ""
            
            # Stream response using simple streaming
            response_content = ""
            tool_usage = ""
            
            async for event in self.agent.stream_message(message, self.session_id):
                event_type = event.get("type", "unknown")
                content = event.get("content", "")
                metadata = event.get("metadata", {})
                
                if event_type == "thinking":
                    # Agent is thinking
                    working_history[-1] = {"role": "assistant", "content": content}
                    yield working_history, ""
                    
                elif event_type == "tool_start":
                    # Tool is starting
                    tool_name = metadata.get("tool_name", "unknown")
                    tool_usage += f"\n\n{content}"
                    working_history[-1] = {"role": "assistant", "content": response_content + tool_usage}
                    yield working_history, ""
                    
                elif event_type == "tool_end":
                    # Tool completed
                    tool_usage += f"\n{content}"
                    working_history[-1] = {"role": "assistant", "content": response_content + tool_usage}
                    yield working_history, ""
                    
                elif event_type == "content":
                    # Stream content from response
                    response_content += content
                    working_history[-1] = {"role": "assistant", "content": response_content + tool_usage}
                    yield working_history, ""
                    
                elif event_type == "error":
                    # Error occurred
                    error_msg = f"\n{content}"
                    working_history[-1] = {"role": "assistant", "content": response_content + tool_usage + error_msg}
                    yield working_history, ""
            
        except Exception as e:
            self.debug_streamer.error(f"Error in stream chat: {e}")
            error_msg = f"❌ **Error streaming message: {str(e)}**"
            working_history[-1] = {"role": "assistant", "content": error_msg}
            yield working_history, ""
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        # Ensure Gradio can serve local static resources via /gradio_api/file=
        RESOURCES_DIR = Path(__file__).parent.parent / "resources"
        try:
            existing_allowed = os.environ.get("GRADIO_ALLOWED_PATHS", "")
            parts = [p for p in existing_allowed.split(os.pathsep) if p]
            if str(RESOURCES_DIR) not in parts:
                parts.append(str(RESOURCES_DIR))
            os.environ["GRADIO_ALLOWED_PATHS"] = os.pathsep.join(parts)
            print(f"Gradio static allowed paths: {os.environ['GRADIO_ALLOWED_PATHS']}")
        except Exception as _e:
            print(f"Warning: could not set GRADIO_ALLOWED_PATHS: {_e}")

        # External CSS file
        css_file_path = Path(__file__).parent.parent / "resources" / "css" / "cmw_copilot_theme.css"

        with gr.Blocks(
            css_paths=[css_file_path],
            title="Analyst Copilot",
            theme=gr.themes.Soft()
        ) as demo:
            
            # Header
            gr.Markdown("# Analyst Copilot", elem_classes=["hero-title"]) 
                       
            with gr.Tabs():
                
                # Chat Tab
                with gr.TabItem("💬 Chat", id="chat"):
                    
                    with gr.Row():
                        with gr.Column(elem_classes=["chat-hints"]):
                            gr.Markdown("""
                            ## 💬 Welcome!
                                                      
                            The Comindware Analyst Copilot focuses on the **Comindware Platform** entity operations (applications, templates, attributes) and uses deterministic tools to execute precise changes.

                            - **Platform Operations First**: Validates your intent and executes tools for entity changes (e.g., create/edit attributes)
                            - **Multi-Model Orchestration**: Tries multiple LLM providers with intelligent fallback
                            - **Compact Structured Output**: Intent → Plan → Validate → Execute → Result
                            """, elem_classes=["chat-hints"]) 
                        with gr.Column(elem_classes=["chat-hints"]):
                            gr.Markdown("""
                            ## ❓ Try asking:
                            
                            - List all applications in the Platform
                            - List all record templates in app 'ERP'
                            - List all attributes in template 'Counterparties', app 'ERP'
                            - Create plain text attribute 'Comment', app 'HR', template 'Candidates'
                            - Create 'Customer ID' text attribute, app 'ERP', template 'Counterparties', custom input mask: ([0-9]{10}|[0-9]{12})
                            - For attribute 'Contact Phone' in app 'CRM', template 'Leads', change display format to Russian phone
                            - Fetch attribute: system name 'Comment', app 'HR', template 'Candidates'
                            - Archive/unarchive attribute, system name 'Comment', app 'HR', template 'Candidates'
                            """, elem_classes=["chat-hints"]) 
                            
                    with gr.Row():
                        with gr.Column(scale=3, elem_classes=["sidebar-card"]):
                            # Chat interface with metadata support for thinking transparency
                            chatbot = gr.Chatbot(
                                label="Chat with the Agent",
                                height=500,
                                show_label=True,
                                container=True,
                                show_copy_button=True,
                                type="messages",
                                elem_id="chatbot-main",
                                elem_classes=["chatbot-card"]
                            )
                            
                            with gr.Row():
                                msg = gr.Textbox(
                                    label="Your Message",
                                    placeholder="Type your message here...",
                                    lines=2,
                                    scale=4,
                                    max_lines=4,
                                    elem_id="message-input",
                                    elem_classes=["message-card"]
                                )
                                with gr.Column():
                                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["cmw-button"])
                                    clear_btn = gr.Button("Clear Chat", variant="secondary", elem_classes=["cmw-button"])
                        
                        with gr.Column(scale=1):
                            with gr.Column(elem_classes=["model-card"]):
                            # Combined status
                                gr.Markdown("### 🤖 Status")
                                status_display = gr.Markdown("🟡 Initializing...")
                                
                            # Quick actions
                            with gr.Column(elem_classes=["quick-actions-card"]):
                                gr.Markdown("### ⚡ Quick Actions")
                                with gr.Column():
                                    quick_list_apps_btn = gr.Button("🔎 List all apps", elem_classes=["cmw-button"])
                                    quick_create_attr_btn = gr.Button("🧩 Create text attribute", elem_classes=["cmw-button"]) 
                                    quick_edit_mask_btn = gr.Button("🛠️ Edit phone mask", elem_classes=["cmw-button"]) 
                                    quick_math_btn = gr.Button("🧮 15 * 23 + 7 = ?", elem_classes=["cmw-button"]) 
                                    quick_code_btn = gr.Button("💻 Python prime check function", elem_classes=["cmw-button"]) 
                                    quick_explain_btn = gr.Button("💭 Explain ML briefly", elem_classes=["cmw-button"]) 
                    
                    # Event handlers
                    def clear_chat():
                        return self.clear_conversation()
                                      
                    def stream_message(message: str, history: List[Dict[str, str]]):
                        """Stream a message to the agent"""
                        if not message.strip():
                            yield history, ""
                            return
                        
                        # Run async generator
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            async def async_stream():
                                async for result in self.stream_chat_with_agent(message, history):
                                    yield result
                            
                            # Convert async generator to regular generator
                            async_gen = async_stream()
                            while True:
                                try:
                                    result = loop.run_until_complete(async_gen.__anext__())
                                    yield result
                                except StopAsyncIteration:
                                    break
                        finally:
                            loop.close()
                    
                    def quick_create_attr():
                        return (
                            "Draft a plan to CREATE a text attribute 'Customer ID' in application 'ERP', template 'Counterparties' "
                            "with display_format=CustomMask and mask ([0-9]{10}|[0-9]{12}), system_name=CustomerID. "
                            "Provide Intent, Plan, Validate, and a DRY-RUN payload preview (compact JSON) for the tool call, "
                            "but DO NOT execute any changes yet. Wait for my confirmation."
                        )
                    
                    def quick_edit_mask():
                        return (
                            "Prepare a safe EDIT plan for attribute 'Contact Phone' (system_name=ContactPhone) in application 'CRM', template 'Leads' "
                            "to change display_format to PhoneRuMask. Provide Intent, Plan, Validate checklist (risk notes), and a DRY-RUN payload preview. "
                            "Do NOT execute changes yet—await my approval."
                        )
                    
                    def quick_list_apps():
                        return (
                            "List all applications in the Platform. "
                            "Format nicely using Markdown. "
                            "Show system names and descriptions if any."
                        )
                    
                    def quick_math():
                        return "What is 15 * 23 + 7? Please show your work step by step."
                    
                    def quick_code():
                        return "Write a Python function to check if a number is prime. Include tests."
                    
                    def quick_explain():
                        return "Explain the concept of machine learning in simple terms."
                    
                    # Connect event handlers
                    send_btn.click(
                        fn=stream_message,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    msg.submit(
                        fn=stream_message,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    clear_btn.click(
                        fn=clear_chat,
                        outputs=[chatbot, msg]
                    )
                                       
                    quick_math_btn.click(
                        fn=quick_math,
                        outputs=[msg]
                    )
                    
                    quick_code_btn.click(
                        fn=quick_code,
                        outputs=[msg]
                    )
                    
                    quick_explain_btn.click(
                        fn=quick_explain,
                        outputs=[msg]
                    )
                    
                    quick_create_attr_btn.click(
                        fn=quick_create_attr,
                        outputs=[msg]
                    )
                    
                    quick_edit_mask_btn.click(
                        fn=quick_edit_mask,
                        outputs=[msg]
                    )
                    
                    quick_list_apps_btn.click(
                        fn=quick_list_apps,
                        outputs=[msg]
                    )
                
                # Logs Tab
                with gr.TabItem("📜 Logs", id="logs"):
                    gr.Markdown("### Initialization Logs")
                    logs_display = gr.Markdown(
                        "🟡 Starting initialization...",
                        elem_classes=["status-card"]
                    )
                    
                    refresh_logs_btn = gr.Button("🔄 Refresh Logs", elem_classes=["cmw-button"])
                    
                    def refresh_logs():
                        return self.get_initialization_logs()
                    
                    def clear_logs():
                        self.log_handler.clear_logs()
                        return "Logs cleared."
                    
                    refresh_logs_btn.click(
                        fn=refresh_logs,
                        outputs=[logs_display]
                    )
                    
                    clear_logs_btn = gr.Button("🗑️ Clear Logs", elem_classes=["cmw-button"])
                    clear_logs_btn.click(
                        fn=clear_logs,
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
            
            # Timer for auto-refresh
            status_timer = gr.Timer(2.0, active=True)
            status_timer.tick(
                fn=update_status,
                outputs=[status_display]
            )
            
            # Auto-refresh logs every 3 seconds
            logs_timer = gr.Timer(3.0, active=True)
            logs_timer.tick(
                fn=refresh_logs,
                outputs=[logs_display]
            )
            
            # Load initial logs
            demo.load(
                fn=refresh_logs,
                outputs=[logs_display]
            )
        
        return demo


# Global demo variable for Gradio reload functionality
demo = None

def get_demo():
    """Get or create the demo interface"""
    global demo
    if demo is None:
        app = NextGenApp()
        demo = app.create_interface()
    return demo

# Initialize demo for Gradio reload functionality
# This ensures demo is available for Gradio's reload mechanism
demo = get_demo()

def main():
    """Main function to run the application"""
    print("🚀 Starting LangChain-Native LLM Agent App...")
    
    # Get the demo interface
    demo = get_demo()
    
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

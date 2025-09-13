"""
LangChain-Native Gradio App - Modularized
=========================================

A modern Gradio application using pure LangChain patterns with modular tab architecture.
This version uses separate tab modules for better organization and maintainability.

Key Features:
- Modular tab architecture (Chat, Logs, Stats)
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
    from agent_ng.tabs import ChatTab, LogsTab, StatsTab
except ImportError:
    # Fallback to relative imports (when running as module)
    try:
        from .langchain_agent import CmwAgent as NextGenAgent, ChatMessage, get_agent_ng
        from .llm_manager import get_llm_manager
        from .debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
        from .streaming_chat import get_chat_interface
        from .tabs import ChatTab, LogsTab, StatsTab
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
        ChatTab = None
        LogsTab = None
        StatsTab = None


class NextGenApp:
    """LangChain-native Gradio application with modular tab architecture"""
    
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
        
        # Initialize tab modules
        self.tabs = {}
        self.components = {}
        
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
    
    def _create_event_handlers(self) -> Dict[str, Any]:
        """Create event handlers for all tabs"""
        return {
            # Chat handlers
            "stream_message": self._stream_message_wrapper,
            "clear_chat": self.clear_conversation,
            
            # Logs handlers
            "refresh_logs": self.get_initialization_logs,
            "clear_logs": self._clear_logs,
            
            # Stats handlers
            "refresh_stats": self._refresh_stats,
            "update_status": self.get_agent_status,
            
            # Quick action handlers
            "quick_math": self._quick_math,
            "quick_code": self._quick_code,
            "quick_explain": self._quick_explain,
            "quick_create_attr": self._quick_create_attr,
            "quick_edit_mask": self._quick_edit_mask,
            "quick_list_apps": self._quick_list_apps,
        }
    
    def _stream_message_wrapper(self, message: str, history: List[Dict[str, str]]):
        """Stream a message to the agent (synchronous wrapper)"""
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
    
    def _clear_logs(self) -> str:
        """Clear logs and return confirmation"""
        self.log_handler.clear_logs()
        return "Logs cleared."
    
    def _refresh_stats(self) -> str:
        """Refresh and return agent statistics"""
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
    
    # Quick action methods
    def _quick_math(self) -> str:
        """Generate math quick action message"""
        return "What is 15 * 23 + 7? Please show your work step by step."
    
    def _quick_code(self) -> str:
        """Generate code quick action message"""
        return "Write a Python function to check if a number is prime. Include tests."
    
    def _quick_explain(self) -> str:
        """Generate explain quick action message"""
        return "Explain the concept of machine learning in simple terms."
    
    def _quick_create_attr(self) -> str:
        """Generate create attribute quick action message"""
        return (
            "Draft a plan to CREATE a text attribute 'Customer ID' in application 'ERP', template 'Counterparties' "
            "with display_format=CustomMask and mask ([0-9]{{10}}|[0-9]{{12}}), system_name=CustomerID. "
            "Provide Intent, Plan, Validate, and a DRY-RUN payload preview (compact JSON) for the tool call, "
            "but DO NOT execute any changes yet. Wait for my confirmation."
        )
    
    def _quick_edit_mask(self) -> str:
        """Generate edit mask quick action message"""
        return (
            "Prepare a safe EDIT plan for attribute 'Contact Phone' (system_name=ContactPhone) in application 'CRM', template 'Leads' "
            "to change display_format to PhoneRuMask. Provide Intent, Plan, Validate checklist (risk notes), and a DRY-RUN payload preview. "
            "Do NOT execute changes yet—await my approval."
        )
    
    def _quick_list_apps(self) -> str:
        """Generate list apps quick action message"""
        return (
            "List all applications in the Platform. "
            "Format nicely using Markdown. "
            "Show system names and descriptions if any."
        )
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface using modular tabs"""
        
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
            title="Comindware Analyst Copilot",
            theme=gr.themes.Soft()
        ) as demo:
            
            # Header
            gr.Markdown("# Analyst Copilot", elem_classes=["hero-title"]) 
                       
            with gr.Tabs():
                # Create event handlers
                event_handlers = self._create_event_handlers()
                
                # Create tabs using modular components
                if ChatTab:
                    chat_tab, chat_components = ChatTab(event_handlers).create_tab()
                    self.components.update(chat_components)
                
                if LogsTab:
                    logs_tab, logs_components = LogsTab(event_handlers).create_tab()
                    self.components.update(logs_components)
                
                if StatsTab:
                    stats_tab, stats_components = StatsTab(event_handlers).create_tab()
                    self.components.update(stats_components)
            
            # Auto-refresh timers
            self._setup_auto_refresh(demo)
        
        return demo
    
    def _setup_auto_refresh(self, demo: gr.Blocks):
        """Setup auto-refresh timers for status and logs"""
        # Status auto-refresh
        if "status_display" in self.components:
            status_timer = gr.Timer(2.0, active=True)
            status_timer.tick(
                fn=self.get_agent_status,
                outputs=[self.components["status_display"]]
            )
        
        # Logs auto-refresh
        if "logs_display" in self.components:
            logs_timer = gr.Timer(3.0, active=True)
            logs_timer.tick(
                fn=self.get_initialization_logs,
                outputs=[self.components["logs_display"]]
            )
        
        # Load initial logs
        if "logs_display" in self.components:
            demo.load(
                fn=self.get_initialization_logs,
                outputs=[self.components["logs_display"]]
            )


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

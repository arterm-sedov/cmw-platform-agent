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
    # from agent_ng.streaming_chat import get_chat_interface  # Module moved to .unused
    from agent_ng.tabs import ChatTab, LogsTab, StatsTab
    from agent_ng.ui_manager import get_ui_manager
    print("✅ Successfully imported all modules using absolute imports")
except ImportError as e1:
    print(f"⚠️ Absolute imports failed: {e1}")
    # Fallback to relative imports (when running as module)
    try:
        from .langchain_agent import CmwAgent as NextGenAgent, ChatMessage, get_agent_ng
        from .llm_manager import get_llm_manager
        from .debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
        # from .streaming_chat import get_chat_interface  # Module moved to .unused
        from .tabs import ChatTab, LogsTab, StatsTab
        from .ui_manager import get_ui_manager
        print("✅ Successfully imported all modules using relative imports")
    except ImportError as e2:
        print(f"❌ Both absolute and relative imports failed:")
        print(f"   Absolute: {e1}")
        print(f"   Relative: {e2}")
        print("⚠️ Running with fallback implementations - some features may not work")
        # Set defaults to prevent further errors
        NextGenAgent = None
        ChatMessage = None
        get_agent_ng = lambda: None
        get_llm_manager = lambda: None
        get_debug_streamer = lambda x: None
        get_log_handler = lambda x: None
        LogLevel = None
        LogCategory = None
        # get_chat_interface = lambda x: None  # Dead code - never used
        ChatTab = None
        LogsTab = None
        StatsTab = None
        get_ui_manager = lambda: None


class NextGenApp:
    """LangChain-native Gradio application with modular tab architecture"""
    
    def __init__(self):
        self.agent: Optional[NextGenAgent] = None
        self.llm_manager = get_llm_manager()
        self.initialization_logs = []
        self.is_initializing = False
        self.initialization_complete = False
        self.session_id = "default"  # LangChain session management
        self._ui_update_needed = False
        
        # Initialize debug system
        self.debug_streamer = get_debug_streamer("app_ng")
        self.log_handler = get_log_handler("app_ng")
        # self.chat_interface = get_chat_interface("app_ng")  # Dead code - never used
        
        # Initialize UI manager with error handling
        try:
            self.ui_manager = get_ui_manager()
            if not self.ui_manager:
                raise ValueError("UI Manager not available")
        except Exception as e:
            print(f"❌ Failed to initialize UI Manager: {e}")
            self.ui_manager = None
        
        # Initialize tab modules
        self.tabs = {}
        self.tab_instances = {}  # Store tab instances for event handlers
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
                
                # Update agent reference in tab instances
                if 'stats' in self.tab_instances:
                    self.tab_instances['stats'].set_agent(self.agent)
                
                # Trigger UI update after agent is ready
                self._trigger_ui_update()
            else:
                self.debug_streamer.error("Agent initialization timeout", LogCategory.INIT)
                self.initialization_logs.append("❌ Agent initialization timeout")
                
        except Exception as e:
            self.debug_streamer.error(f"Initialization failed: {str(e)}", LogCategory.INIT)
            self.initialization_logs.append(f"❌ Initialization failed: {str(e)}")
        
        self.is_initializing = False
    
    
    
    def is_ready(self) -> bool:
        """Check if the app is ready (LangChain-native pattern)"""
        return self.initialization_complete and self.agent is not None and self.agent.is_ready()
    
    def clear_conversation(self) -> Tuple[List[Dict[str, str]], str]:
        """Clear the conversation history (LangChain-native pattern)"""
        if self.agent:
            self.agent.clear_conversation(self.session_id)
            # Start new conversation tracking for token counting
            if hasattr(self.agent, 'token_tracker'):
                self.agent.token_tracker.start_new_conversation()
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
        
        # Track execution time
        start_time = time.time()
        
        try:
            self.debug_streamer.info(f"Streaming message: {message[:50]}...")
            
            # Add user message to history
            working_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            
            # Get prompt token count for user message (will be displayed below assistant response)
            prompt_tokens = None
            if self.agent:
                try:
                    prompt_tokens = self.agent.count_prompt_tokens_for_chat(history, message)
                except Exception as e:
                    self.debug_streamer.warning(f"Failed to get prompt token count: {e}")
            
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
                    tool_usage += f"{content}"
                    working_history[-1] = {"role": "assistant", "content": response_content + tool_usage}
                    yield working_history, ""
                    
                elif event_type == "tool_end":
                    # Tool completed
                    tool_usage += f"{content}"
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
            
            # Add API token count to final response
            # Add token counts below assistant response
            token_displays = []
            
            # Add prompt tokens if available
            if prompt_tokens:
                token_displays.append(f"**Prompt tokens:** {prompt_tokens.formatted}")
            
            # Add API tokens if available
            if self.agent:
                try:
                    print(f"🔍 DEBUG: Getting last API tokens from agent")
                    last_api_tokens = self.agent.get_last_api_tokens()
                    print(f"🔍 DEBUG: Last API tokens: {last_api_tokens}")
                    if last_api_tokens:
                        token_displays.append(f"**API tokens:** {last_api_tokens.formatted}")
                        print(f"🔍 DEBUG: Added API token display")
                    else:
                        print("🔍 DEBUG: No API tokens available")
                except Exception as e:
                    print(f"🔍 DEBUG: API token error: {e}")
                    self.debug_streamer.warning(f"Failed to get API token count: {e}")
            
            # Add execution time
            execution_time = time.time() - start_time
            token_displays.append(f"**Execution time:** {execution_time:.2f}s")
            
            # Combine all token displays
            if token_displays:
                token_display = "\n\n" + "\n".join(token_displays)
                working_history[-1] = {"role": "assistant", "content": response_content + tool_usage + token_display}
                print(f"🔍 DEBUG: Added token display: {token_display}")
            
            # Final yield with updated stats
            yield working_history, ""
            
        except Exception as e:
            self.debug_streamer.error(f"Error in stream chat: {e}")
            execution_time = time.time() - start_time
            error_msg = f"❌ **Error streaming message: {str(e)}**\n\n**Execution time:** {execution_time:.2f}s"
            working_history[-1] = {"role": "assistant", "content": error_msg}
            yield working_history, ""
    
    def _create_event_handlers(self) -> Dict[str, Any]:
        """Create event handlers for all tabs"""
        return {
            # Chat handlers (core functionality)
            "stream_message": self._stream_message_wrapper,
            "clear_chat": self.clear_conversation,
            
            # Status and monitoring handlers
            "update_status": self._update_status,
            "refresh_logs": self._refresh_logs,
            "refresh_stats": self._refresh_stats,
            "update_all_ui": self.update_all_ui_components,
            "trigger_ui_update": self.trigger_ui_update,
        }
    
    def _stream_message_wrapper(self, message: str, history: List[Dict[str, str]]):
        """Stream a message to the agent (synchronous wrapper)"""
        if not message.strip():
            yield history, ""
            return
        
        # Use the existing event loop or create a new one
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, need to run in a thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    async def async_stream():
                        async for result in self.stream_chat_with_agent(message, history):
                            yield result
                    
                    # Convert async generator to regular generator
                    async_gen = async_stream()
                    results = []
                    while True:
                        try:
                            result = new_loop.run_until_complete(async_gen.__anext__())
                            results.append(result)
                        except StopAsyncIteration:
                            break
                    return results
                finally:
                    new_loop.close()
            
            # Run in thread and collect all results
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                results = future.result()
                
            # Yield all results
            for result in results:
                yield result
                
        except RuntimeError:
            # No event loop running, create a new one
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
        
        # Refresh UI after streaming completes (EVENT-DRIVEN)
        self._refresh_ui_after_message()
    
    def _update_status(self) -> str:
        """Update status display - delegates to stats tab"""
        stats_tab = self.tab_instances.get('stats')
        if stats_tab and hasattr(stats_tab, 'format_stats_display'):
            return stats_tab.format_stats_display()
        
        # Fallback status
        if self.agent and self.agent.is_ready():
            return "✅ **Agent Ready**"
        else:
            return "🟡 **Initializing...**"
    
    def _refresh_logs(self) -> str:
        """Refresh logs display - delegates to logs tab"""
        logs_tab = self.tab_instances.get('logs')
        if logs_tab and hasattr(logs_tab, 'get_initialization_logs'):
            return logs_tab.get_initialization_logs()
        
        # Fallback logs
        return "\n".join(self.initialization_logs) if self.initialization_logs else "No logs available"
    
    def _refresh_stats(self) -> str:
        """Refresh stats display - delegates to stats tab"""
        stats_tab = self.tab_instances.get('stats')
        if stats_tab and hasattr(stats_tab, 'format_stats_display'):
            return stats_tab.format_stats_display()
        
        # Fallback stats
        if self.agent and self.agent.is_ready():
            return "✅ Agent Ready"
        else:
            return "🟡 Agent Initializing..."
    
    def _trigger_ui_update(self):
        """Trigger UI update after agent initialization or message processing"""
        try:
            print("🔍 DEBUG: Triggering UI update...")
            # Store the update trigger - the UI will check this
            self._ui_update_needed = True
        except Exception as e:
            print(f"🔍 DEBUG: Error triggering UI update: {e}")
    
    def check_and_clear_ui_update(self) -> bool:
        """Check if UI update is needed and clear the flag"""
        if self._ui_update_needed:
            self._ui_update_needed = False
            return True
        return False
    
    def update_all_ui_components(self) -> Tuple[str, str, str]:
        """Update all UI components and return their values"""
        status = self._update_status()
        stats = self._refresh_stats()
        logs = self._refresh_logs()
        return status, stats, logs
    
    def trigger_ui_update(self):
        """Trigger UI update after agent initialization or message processing"""
        try:
            print("🔍 DEBUG: Triggering UI update...")
            # This will be called when the agent is ready or after messages
            # The actual UI update will happen through Gradio's event system
            self._ui_update_needed = True
        except Exception as e:
            print(f"🔍 DEBUG: Error triggering UI update: {e}")
    
    def _refresh_ui_after_message(self):
        """Refresh all UI components after a message is processed (EVENT-DRIVEN)"""
        try:
            # Trigger UI update after message processing
            self.trigger_ui_update()
            
            print("🔍 DEBUG: UI refreshed after message completion (EVENT-DRIVEN)")
            
        except Exception as e:
            print(f"🔍 DEBUG: Error refreshing UI after message: {e}")
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface using UI Manager and modular tabs"""
        # Validate UI Manager
        if not self.ui_manager:
            raise RuntimeError("UI Manager not available - cannot create interface")
        
        # Create event handlers
        event_handlers = self._create_event_handlers()
        
        # Create tab modules with error handling
        tab_modules = []
        try:
            if ChatTab:
                chat_tab = ChatTab(event_handlers)
                tab_modules.append(chat_tab)
                self.tab_instances['chat'] = chat_tab
            else:
                print("⚠️ ChatTab not available")
                
            if LogsTab:
                logs_tab = LogsTab(event_handlers)
                logs_tab.set_main_app(self)  # Pass main app reference
                tab_modules.append(logs_tab)
                self.tab_instances['logs'] = logs_tab
            else:
                print("⚠️ LogsTab not available")
                
            if StatsTab:
                stats_tab = StatsTab(event_handlers)
                stats_tab.set_agent(self.agent)  # Pass agent reference
                tab_modules.append(stats_tab)
                self.tab_instances['stats'] = stats_tab
            else:
                print("⚠️ StatsTab not available")
        except Exception as e:
            print(f"❌ Error creating tab modules: {e}")
            raise
        
        # Use UI Manager to create interface
        try:
            demo = self.ui_manager.create_interface(tab_modules, event_handlers)
        except Exception as e:
            print(f"❌ Error creating interface: {e}")
            raise
        
        # Consolidate all components from UI Manager (single source of truth)
        self.components = self.ui_manager.get_components()
        
        # Set agent reference on all tabs that support it
        if self.agent:
            self.ui_manager.set_agent(self.agent)
        
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

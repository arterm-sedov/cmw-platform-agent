"""
Stats Tab Module for App NG
==========================

Handles statistics monitoring, display, and management functionality.
This module encapsulates all statistics-related UI components and functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

from collections.abc import Callable
import logging
import time
from typing import Any, Dict, Optional, Tuple

import gradio as gr


class StatsTab:
    """Stats tab component for statistics and monitoring"""

    def __init__(self, event_handlers: dict[str, Callable], language: str = "en", i18n_instance: gr.I18n | None = None):
        self.event_handlers = event_handlers
        self.components = {}
        self.agent = None  # Will be set by the app
        self.main_app = None  # Reference to main app for session management
        self._last_conversation_stats = None  # Track last stats for change detection
        self.language = language
        self.i18n = i18n_instance

    def create_tab(self) -> tuple[gr.TabItem, dict[str, Any]]:
        """
        Create the stats tab with all its components.

        Returns:
            Tuple of (TabItem, components_dict)
        """
        logging.getLogger(__name__).info("✅ StatsTab: Creating stats interface...")

        with gr.TabItem(self._get_translation("tab_stats"), id="stats") as tab:
            # Create stats interface
            self._create_stats_interface()

            # Connect event handlers
            self._connect_events()

        logging.getLogger(__name__).info("✅ StatsTab: Successfully created with all components and event handlers")
        return tab, self.components

    def _create_stats_interface(self):
        """Create the statistics monitoring interface"""
        # Main stats display area - similar to logs interface
        with gr.Column(scale=1, min_width=400):
            self.components["stats_display"] = gr.Textbox(
                value=self._get_translation("stats_loading"),
                label=self._get_translation("stats_title"),
                lines=20,
                max_lines=30,
                interactive=False,
                show_copy_button=True,
                container=True,
                elem_id="stats-display"
            )

        # Control buttons row
        with gr.Row(equal_height=True):
            self.components["refresh_stats_btn"] = gr.Button(
                self._get_translation("refresh_stats_button"),
                elem_classes=["cmw-button"],
                scale=1
            )
            self.components["clear_stats_btn"] = gr.Button(
                self._get_translation("clear_stats_button"),
                elem_classes=["cmw-button"],
                scale=1
            )

    def _connect_events(self):
        """Connect all event handlers for the stats tab with concurrency control"""
        logging.getLogger(__name__).debug("🔗 StatsTab: Connecting event handlers with concurrency control...")

        # Get queue manager for concurrency control
        queue_manager = getattr(self, "main_app", None)
        if queue_manager:
            queue_manager = getattr(queue_manager, "queue_manager", None)

        if queue_manager:
            # Apply concurrency settings to stats events
            from agent_ng.queue_manager import apply_concurrency_to_click_event

            refresh_config = apply_concurrency_to_click_event(
                queue_manager, "stats_refresh", self.refresh_stats,
                [], [self.components["stats_display"]]
            )
            self.components["refresh_stats_btn"].click(**refresh_config)

            clear_config = apply_concurrency_to_click_event(
                queue_manager, "stats_clear", self.clear_stats,
                [], [self.components["stats_display"]]
            )
            self.components["clear_stats_btn"].click(**clear_config)
        else:
            # Fallback to default behavior
            logging.getLogger(__name__).warning("⚠️ Queue manager not available - using default stats configuration")
            self.components["refresh_stats_btn"].click(
                fn=self.refresh_stats,
                outputs=[self.components["stats_display"]]
            )

            self.components["clear_stats_btn"].click(
                fn=self.clear_stats,
                outputs=[self.components["stats_display"]]
            )

        logging.getLogger(__name__).debug("✅ StatsTab: All event handlers connected successfully")

    def get_components(self) -> dict[str, Any]:
        """Get all components created by this tab"""
        return self.components

    def get_stats_display_component(self) -> gr.Textbox:
        """Get the stats display component for auto-refresh"""
        return self.components["stats_display"]

    def set_agent(self, agent):
        """Set the agent reference for stats access (deprecated - use session-specific agents)"""
        # This method is kept for compatibility but is no longer used
        # Stats are now accessed through session-specific agents

    def set_main_app(self, app):
        """Set reference to main app for session management"""
        self.main_app = app

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        # Always use direct translation for now to avoid i18n metadata issues
        from ..i18n_translations import get_translation_key
        return get_translation_key(key, self.language)

    def format_stats_display(self, request: gr.Request = None) -> str:
        """Format and return the complete stats display - always session-aware"""
        # Try to get request from Gradio context if not provided
        if not request:
            try:
                import gradio as gr
                from gradio.context import Context
                if hasattr(Context, "root_block") and Context.root_block:
                    request = Context.root_block.get_request()
            except:
                pass

        # Get session-specific agent
        agent = None
        if request and hasattr(self, "main_app") and hasattr(self.main_app, "session_manager"):
            session_id = self.main_app.session_manager.get_session_id(request)
            agent = self.main_app.session_manager.get_session_agent(session_id)
        elif hasattr(self, "main_app") and hasattr(self.main_app, "session_manager"):
            # For auto-refresh, show a generic message since we can't determine the session
            return self._get_translation("stats_auto_refresh_message")

        # No fallback to global agent - use session-specific agents only

        # Format stats with the appropriate agent
        if not agent:
            return self._get_translation("agent_not_available")

        try:
            # Get basic agent statistics
            stats = agent.get_stats()

            # Format complete display using existing translation resources exactly as they are
            return f"""
{self._get_translation('agent_status_section')}
- {self._get_translation('status_ready_true' if stats['agent_status']['is_ready'] else 'status_ready_false')}
- LLM: {stats['llm_info'].get('model_name', 'Unknown')}
- {self._get_translation('provider_info').format(provider=stats['llm_info'].get('provider', 'Unknown'))}

{self._get_translation('conversation_section')}
- {self._get_translation('messages_label')}: {stats['conversation_stats']['message_count']}
- {self._get_translation('user_messages_label')}: {stats['conversation_stats']['user_messages']}
- {self._get_translation('assistant_messages_label')}: {stats['conversation_stats']['assistant_messages']}
- {self._get_translation('total_messages_label')}: {stats['conversation_stats']['message_count']}

{self._get_translation('tools_section')}
- {self._get_translation('available_label')}: {stats['agent_status']['tools_count']}{self._format_tools_stats(agent)}
            """
        except Exception as e:
            return f"{self._get_translation('error_loading_stats')}: {e!s}"




    def _format_tools_stats(self, agent=None) -> str:
        """Format tools usage statistics from session-specific agent"""
        if not agent:
            return ""

        try:
            # Get tool usage from memory manager
            if hasattr(agent, "memory_manager") and agent.memory_manager:
                total_tool_calls = 0
                unique_tools = set()

                for _conversation_id, conversation in agent.memory_manager.memories.items():
                    for message in conversation:
                        if hasattr(message, "type") and message.type == "tool":
                            total_tool_calls += 1
                            # Try to get tool name from message
                            if hasattr(message, "name"):
                                unique_tools.add(message.name)

                if total_tool_calls > 0:
                    return f"\n- {self._get_translation('used_label')}: {len(unique_tools)} {self._get_translation('unique_tools_label')}\n- {self._get_translation('total_calls_label')}: {total_tool_calls}"
                else:
                    return f"\n- {self._get_translation('used_label')}: 0 {self._get_translation('unique_tools_label')}"
            else:
                return ""
        except Exception:
            return ""

    # Stats handler methods
    def refresh_stats(self, request: gr.Request = None) -> str:
        """Refresh and return agent statistics"""
        return self.format_stats_display(request)

    def clear_stats(self, request: gr.Request = None) -> str:
        """Clear stats and return confirmation"""
        return self._get_translation("stats_cleared")

    def get_agent_status(self, agent=None) -> str:
        """Get current agent status with comprehensive details from session-specific agent"""
        if not agent:
            return self._get_translation("agent_initializing")

        status = agent.get_status()
        if status["is_ready"]:
            llm_info = agent.get_llm_info()
            return f"""{self._get_translation('agent_status_ready')}

{self._get_translation('provider_info').format(provider=llm_info.get('provider', 'Unknown'))}

{self._get_translation('model_info').format(model=llm_info.get('model_name', 'Unknown'))}

{self._get_translation('status_label')} {self._get_translation('healthy_status') if llm_info.get('is_healthy', False) else self._get_translation('unhealthy_status')}

{self._get_translation('tools_label')} {status['tools_count']} {self._get_translation('available_label')}

{self._get_translation('last_used_label')} {time.ctime(llm_info.get('last_used', 0))}"""
        else:
            return f"{self._get_translation('agent_status_initializing')}\n\n{self._get_translation('status_label')} {status.get('status', 'Unknown')}\n{self._get_translation('tools_label')} {status.get('tools_count', 0)} {self._get_translation('available_label')}"

"""
Agent Configuration
==================

Central configuration file for the CMW Platform Agent.
Contains all configurable settings including refresh intervals, timeouts, and other parameters.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load .env file before reading environment variables
# Find .env file relative to project root (parent of agent_ng directory)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

@dataclass
class RefreshIntervals:
    """Auto-refresh intervals for UI components"""
    interval: float = 15.0     # General UI refresh interval (seconds)
    iteration: float = 2.0     # Iteration/progress refresh interval (seconds)

@dataclass
class AgentSettings:
    """Main agent configuration settings"""
    # Language settings
    default_language: str = "ru"
    supported_languages: list = None

    # Port settings
    default_port: int = 7860
    auto_port_range: int = 10  # Number of ports to try when auto-finding

    # UI settings
    refresh_intervals: RefreshIntervals = None

    # Agent settings
    max_conversation_history: int = 50
    max_tokens_per_request: int = 4000
    request_timeout: float = 30.0

    # LLM provider/model settings
    default_provider: str = "mistral"
    default_model: str = None  # None means use first model (index 0)

    # Debug settings
    debug_mode: bool = False
    verbose_logging: bool = False

    # LangSmith observability settings
    langsmith_tracing: bool = False
    langsmith_project: str = "cmw-agent"

    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.supported_languages is None:
            self.supported_languages = ["en", "ru"]

        if self.refresh_intervals is None:
            self.refresh_intervals = RefreshIntervals()

class AgentConfig:
    """Central configuration manager for the CMW Platform Agent"""

    def __init__(self):
        self.settings = AgentSettings()
        self._load_from_environment()

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Language settings - use GRADIO_DEFAULT_LANGUAGE for Gradio UI
        if os.getenv("GRADIO_DEFAULT_LANGUAGE"):
            self.settings.default_language = os.getenv("GRADIO_DEFAULT_LANGUAGE")

        # Port settings - use GRADIO_DEFAULT_PORT for Gradio server
        if os.getenv("GRADIO_DEFAULT_PORT"):
            try:
                self.settings.default_port = int(os.getenv("GRADIO_DEFAULT_PORT"))
            except ValueError as e:
                logging.warning(f"Invalid GRADIO_DEFAULT_PORT value, using default: {e}")

        # LLM provider/model settings
        if os.getenv("AGENT_PROVIDER"):
            self.settings.default_provider = os.getenv("AGENT_PROVIDER").strip()

        model_env = os.getenv("AGENT_DEFAULT_MODEL")
        if model_env and model_env.strip():
            self.settings.default_model = model_env.strip()

        # Debug settings
        if os.getenv("CMW_DEBUG_MODE", "").lower() in ["true", "1", "yes"]:
            self.settings.debug_mode = True

        if os.getenv("CMW_VERBOSE_LOGGING", "").lower() in ["true", "1", "yes"]:
            self.settings.verbose_logging = True

        # LangSmith settings
        if os.getenv("LANGSMITH_TRACING", "").lower() in ["true", "1", "yes"]:
            self.settings.langsmith_tracing = True

        if os.getenv("LANGSMITH_PROJECT"):
            self.settings.langsmith_project = os.getenv("LANGSMITH_PROJECT")

        # Refresh intervals from environment
        self._load_refresh_intervals_from_env()

    def _load_refresh_intervals_from_env(self):
        """Load refresh intervals from environment variables, if provided.

        Supported variables (seconds, float):
          - UI_REFRESH_INTERVAL          (general UI refresh)
          - ITERATION_REFRESH_INTERVAL   (progress/iteration refresh)

        Falls back to defaults when not set or invalid.
        """
        interval_val = os.getenv("UI_REFRESH_INTERVAL")
        if interval_val is not None:
            try:
                self.settings.refresh_intervals.interval = float(interval_val)
            except ValueError as e:
                logging.warning(f"Invalid UI_REFRESH_INTERVAL value, using default: {e}")

        iteration_val = os.getenv("ITERATION_REFRESH_INTERVAL")
        if iteration_val is not None:
            try:
                self.settings.refresh_intervals.iteration = float(iteration_val)
            except ValueError as e:
                logging.warning(f"Invalid ITERATION_REFRESH_INTERVAL value, using default: {e}")

    def get_refresh_intervals(self) -> RefreshIntervals:
        """Get the current refresh intervals configuration"""
        return self.settings.refresh_intervals

    def get_language_settings(self) -> dict[str, Any]:
        """Get language-related settings"""
        return {
            "default_language": self.settings.default_language,
            "supported_languages": self.settings.supported_languages
        }

    def get_port_settings(self) -> dict[str, Any]:
        """Get port-related settings"""
        return {
            "default_port": self.settings.default_port,
            "auto_port_range": self.settings.auto_port_range
        }

    def get_agent_settings(self) -> dict[str, Any]:
        """Get agent-related settings"""
        return {
            "max_conversation_history": self.settings.max_conversation_history,
            "max_tokens_per_request": self.settings.max_tokens_per_request,
            "request_timeout": self.settings.request_timeout
        }

    def get_llm_settings(self) -> dict[str, Any]:
        """Get LLM provider/model settings"""
        return {
            "default_provider": self.settings.default_provider,
            "default_model": self.settings.default_model
        }

    def get_debug_settings(self) -> dict[str, Any]:
        """Get debug-related settings"""
        return {
            "debug_mode": self.settings.debug_mode,
            "verbose_logging": self.settings.verbose_logging
        }

    def get_langsmith_settings(self) -> dict[str, Any]:
        """Get LangSmith-related settings"""
        return {
            "langsmith_tracing": self.settings.langsmith_tracing,
            "langsmith_project": self.settings.langsmith_project
        }

    def update_setting(self, category: str, key: str, value: Any):
        """Update a specific setting"""
        if category == "refresh_intervals":
            if hasattr(self.settings.refresh_intervals, key):
                setattr(self.settings.refresh_intervals, key, value)
        elif hasattr(self.settings, key):
            setattr(self.settings, key, value)

    def print_config(self):
        """Print current configuration"""
        print("🔧 Agent Configuration:")
        print(f"  Language: {self.settings.default_language}")
        print(f"  Port: {self.settings.default_port}")
        print(f"  Provider: {self.settings.default_provider}")
        if self.settings.default_model:
            print(f"  Model: {self.settings.default_model}")
        else:
            print("  Model: (default - first model)")
        print(f"  Debug Mode: {self.settings.debug_mode}")
        print(f"  LangSmith Tracing: {self.settings.langsmith_tracing}")
        print(f"  LangSmith Project: {self.settings.langsmith_project}")
        print(f"  Refresh Interval: {self.settings.refresh_intervals.interval}s")
        print(f"  Iteration Interval: {self.settings.refresh_intervals.iteration}s")

# Global configuration instance
config = AgentConfig()

# Convenience functions for easy access
def get_refresh_intervals() -> RefreshIntervals:
    """Get refresh intervals configuration"""
    return config.get_refresh_intervals()

def get_language_settings() -> dict[str, Any]:
    """Get language settings"""
    return config.get_language_settings()

def get_port_settings() -> dict[str, Any]:
    """Get port settings"""
    return config.get_port_settings()

def get_agent_settings() -> dict[str, Any]:
    """Get agent settings"""
    return config.get_agent_settings()

def get_debug_settings() -> dict[str, Any]:
    """Get debug settings"""
    return config.get_debug_settings()

def get_langsmith_settings() -> dict[str, Any]:
    """Get LangSmith settings"""
    # Create fresh instance to pick up current environment variables
    fresh_config = AgentConfig()
    return fresh_config.get_langsmith_settings()

def get_llm_settings() -> dict[str, Any]:
    """Get LLM provider/model settings"""
    return config.get_llm_settings()


# Order matches ``NextGenApp.create_interface()`` tab registration (excluding sidebar TabItem).
_UI_TAB_BUILD_ORDER: tuple[str, ...] = (
    "home",
    "chat",
    "logs",
    "stats",
    "config",
    "downloads",
)


def get_ui_tab_allowlist() -> frozenset[str] | None:
    """Subset of UI tabs to build (debug / bisect lazy-tab issues).

    Environment ``CMW_UI_TABS``: comma-separated keys. When non-empty, it wins over
    ``CMW_UI_TAB_LIMIT``.

    Environment ``CMW_UI_TAB_LIMIT``: positive integer — keep only the first *N* tabs in
    ``_UI_TAB_BUILD_ORDER`` (e.g. ``3`` → home, chat, logs). Unset / invalid = no limit.

    Sidebar uses key ``sidebar`` and is only included when listed in ``CMW_UI_TABS`` (never
    via ``CMW_UI_TAB_LIMIT``).

    Keys: ``home``, ``chat``, ``logs``, ``stats``, ``config``, ``downloads``, ``sidebar``.
    """
    raw = (os.getenv("CMW_UI_TABS") or "").strip()
    if raw:
        return frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
    limit_raw = (os.getenv("CMW_UI_TAB_LIMIT") or "").strip()
    if limit_raw.isdigit():
        n = int(limit_raw, 10)
        if n <= 0:
            return None
        keys = _UI_TAB_BUILD_ORDER[: min(n, len(_UI_TAB_BUILD_ORDER))]
        return frozenset(keys)
    return None


def env_flag_true(name: str) -> bool:
    """Return True if ``name`` is set to a truthy token (1/true/yes/on)."""
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def get_ui_stack_home_chat() -> bool:
    """Stack home + chat in one column without ``gr.Tabs`` (eliminates tab-switch mount).

    Environment ``CMW_UI_STACK_HOME_CHAT`` — requires every built tab module to implement ``build_ui()``.
    """
    return env_flag_true("CMW_UI_STACK_HOME_CHAT")


def get_ui_disable_auto_timers() -> bool:
    """Skip ``gr.Timer`` tick handlers (bisect periodic-queue interference)."""
    return env_flag_true("CMW_UI_DISABLE_AUTO_TIMERS")


def get_ui_export_html_after_turn() -> bool:
    """Also build HTML export synchronously after each chat turn (slow; can stall Gradio 6).

    When ``False`` (default), only Markdown is generated on the post-stream path; the HTML
    download stays hidden until you opt in via env.

    Environment ``CMW_UI_EXPORT_HTML_AFTER_TURN``: ``1``/``true``/``yes``/``on`` to enable.
    """
    return env_flag_true("CMW_UI_EXPORT_HTML_AFTER_TURN")


def get_ui_download_prep_after_stream() -> bool:
    """Run download-file preparation chained on ``submit_event`` right after streaming ends.

    When ``False`` (default), preparation runs only when the user selects the Downloads tab
    (fewer simultaneous UI updates after each token — closer to cmw-rag-style tails).

    Environment ``CMW_UI_DOWNLOAD_PREP_AFTER_STREAM``: ``1``/``true``/``yes``/``on`` to enable.
    """
    return env_flag_true("CMW_UI_DOWNLOAD_PREP_AFTER_STREAM")

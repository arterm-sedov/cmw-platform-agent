"""
Dead Code: Configuration/Setup Functions
=======================================

These configuration and setup functions were defined but rarely used or only used in specific contexts.
They can be safely removed unless needed for future configuration management.

Extracted from:
- agent_ng/agent_config.py
- agent_ng/langsmith_config.py
- agent_ng/langfuse_config.py
"""

# From agent_ng/agent_config.py
def print_config(self):
    """Print current configuration"""
    print("🔧 Agent Configuration:")
    print(f"  Language: {self.settings.default_language}")
    print(f"  Port: {self.settings.default_port}")
    print(f"  Debug Mode: {self.settings.debug_mode}")
    print(f"  LangSmith Tracing: {self.settings.langsmith_tracing}")
    print(f"  LangSmith Project: {self.settings.langsmith_project}")
    print(f"  Refresh Intervals:")
    print(f"    Status: {self.settings.refresh_intervals.status}s")
    print(f"    Logs: {self.settings.refresh_intervals.logs}s")
    print(f"    Stats: {self.settings.refresh_intervals.stats}s")
    print(f"    Progress: {self.settings.refresh_intervals.progress}s")

# From agent_ng/langsmith_config.py
def get_openai_wrapper():
    """Get OpenAI wrapper for LangSmith tracing"""
    try:
        from langsmith import Client
        from langsmith.wrappers import wrap_openai

        config = get_langsmith_config()
        if not config.is_configured():
            return None

        client = Client(
            api_key=config.api_key,
            api_url=config.api_url
        )

        return wrap_openai(client)
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: Could not create OpenAI wrapper: {e}")
        return None

# From agent_ng/langfuse_config.py
def get_langfuse_callback_handler():
    """Get Langfuse callback handler if configured"""
    try:
        config = get_langfuse_config()
        if not config.is_configured():
            return None

        from langfuse import Langfuse
        from langfuse.callback import CallbackHandler

        langfuse = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host
        )

        return CallbackHandler(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host
        )
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: Could not create Langfuse callback handler: {e}")
        return None

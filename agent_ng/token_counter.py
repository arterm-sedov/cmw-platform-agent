"""
Token Counter Module
===================

Lean, modular token counting system with abstract design patterns.
Provides precise token counting using tiktoken and real-time API tracking.

Key Features:
- Abstract TokenCounter base class for extensibility
- TiktokenCounter for precise token counting
- ApiTokenCounter for real-time API token tracking
- ConversationTokenTracker for cumulative tracking
- Clean separation of concerns with DRY principles

Usage:
    from token_counter import ConversationTokenTracker

    tracker = ConversationTokenTracker()
    tokens = tracker.count_prompt_tokens(messages)
    tracker.track_llm_response(response, messages)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import time
from typing import Any, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
import tiktoken

from .token_budget import (
    TOKEN_STATUS_CRITICAL,
    TOKEN_STATUS_CRITICAL_THRESHOLD,
    TOKEN_STATUS_GOOD,
    TOKEN_STATUS_MODERATE,
    TOKEN_STATUS_MODERATE_THRESHOLD,
    TOKEN_STATUS_UNKNOWN,
    TOKEN_STATUS_WARNING,
    TOKEN_STATUS_WARNING_THRESHOLD,
    compute_token_budget_snapshot,
)


@dataclass
class TokenCount:
    """Immutable token count data structure"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    is_estimated: bool = False
    source: str = "unknown"

    @property
    def formatted(self) -> str:
        """Format token count for display"""
        if self.is_estimated:
            return f"~{self.total_tokens:,} total (estimated via {self.source})"
        return (
            f"{self.total_tokens:,} total "
            f"({self.input_tokens:,} input + {self.output_tokens:,} output)"
        )


class TokenCounter(ABC):
    """Abstract base class for token counting strategies"""

    @abstractmethod
    def count_tokens(
        self, content: str | list[BaseMessage] | list[dict[str, Any]]
    ) -> TokenCount:
        """Count tokens in content"""
        raise NotImplementedError

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the source name for this counter"""
        raise NotImplementedError


class TiktokenCounter(TokenCounter):
    """Precise token counting using tiktoken"""

    def __init__(self, model: str = "gpt-4") -> None:
        self.encoding = self._init_encoding(model)
        self.model = model

    def _init_encoding(self, model: str) -> tiktoken.Encoding | None:
        """Initialize tiktoken encoding with fallback"""
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            except Exception:
                return None

    def count_tokens(
        self, content: str | list[BaseMessage] | list[dict[str, Any]]
    ) -> TokenCount:
        """Count tokens using tiktoken with LangChain fallback"""
        if not content:
            return TokenCount(
                0, 0, 0, is_estimated=False, source=self.get_source_name()
            )

        try:
            if self.encoding:
                text = self._extract_text(content)
                tokens = self.encoding.encode(text)
                token_count = len(tokens)
                return TokenCount(
                    token_count,
                    0,
                    token_count,
                    is_estimated=False,
                    source=self.get_source_name(),
                )
            # Fallback to LangChain's count_tokens_approximately
            return self._langchain_fallback(content)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Tiktoken encoding failed, falling back to LangChain: %s", exc
            )
            # Final fallback to LangChain
            return self._langchain_fallback(content)

    def _langchain_fallback(
        self, content: str | list[BaseMessage] | list[dict[str, Any]]
    ) -> TokenCount:
        """Fallback to LangChain's count_tokens_approximately"""
        try:
            if isinstance(content, str):
                # Convert string to BaseMessage for LangChain counting
                messages = [HumanMessage(content=content)]
            elif (
                isinstance(content, list)
                and content
                and isinstance(content[0], BaseMessage)
            ):
                messages = content
            else:
                # Convert other formats to BaseMessage
                text = self._extract_text(content)
                messages = [HumanMessage(content=text)]

            estimated_tokens = count_tokens_approximately(messages)
            return TokenCount(
                estimated_tokens,
                0,
                estimated_tokens,
                is_estimated=True,
                source="langchain_estimation",
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "LangChain token counting failed, using character estimation: %s", exc
            )
            # Ultimate fallback to character estimation
            text = self._extract_text(content)
            estimated = len(text) // 4
            return TokenCount(
                estimated,
                0,
                estimated,
                is_estimated=True,
                source="character_estimation",
            )

    def _extract_text(
        self, content: str | list[BaseMessage] | list[dict[str, Any]]
    ) -> str:
        """Extract text from various content formats"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            if content and isinstance(content[0], BaseMessage):
                return "\n".join(
                    msg.content for msg in content if hasattr(msg, "content")
                )
            if content and isinstance(content[0], dict):
                return "\n".join(
                    str(msg.get("content", ""))
                    for msg in content
                    if isinstance(msg, dict) and msg.get("content")
                )
        return str(content)

    def get_source_name(self) -> str:
        return "tiktoken"


class ApiTokenCounter(TokenCounter):
    """Real-time API token counting with tiktoken fallback"""

    def __init__(self, tiktoken_counter: TiktokenCounter) -> None:
        self.tiktoken_counter = tiktoken_counter
        self._last_api_tokens: TokenCount | None = None

    def set_api_tokens(
        self, input_tokens: int, output_tokens: int, total_tokens: int
    ) -> None:
        """Set token count from API response"""
        self._last_api_tokens = TokenCount(
            input_tokens,
            output_tokens,
            total_tokens,
            is_estimated=False,
            source="api",
        )

    def count_tokens(
        self, content: str | list[BaseMessage] | list[dict[str, Any]]
    ) -> TokenCount:
        """Count tokens with API fallback to tiktoken"""
        if self._last_api_tokens:
            return self._last_api_tokens

        # Fallback to tiktoken estimation
        tiktoken_count = self.tiktoken_counter.count_tokens(content)
        return TokenCount(
            tiktoken_count.input_tokens,
            tiktoken_count.output_tokens,
            tiktoken_count.total_tokens,
            is_estimated=True,
            source="tiktoken_estimation",
        )

    def get_source_name(self) -> str:
        return "api_with_tiktoken_fallback"


class ConversationTokenTracker:
    """Cumulative conversation token tracking"""

    def __init__(self):
        self.tiktoken_counter = TiktokenCounter()
        self.api_counter = ApiTokenCounter(self.tiktoken_counter)
        self.conversation_tokens = 0  # Total across all conversations
        # Session-bound tokens (what UI calls "Диалог")
        self.session_tokens = 0
        # Backward-compatible alias for older naming
        self.last_conversation_tokens = self.session_tokens
        self.message_count = 0
        self._last_prompt_tokens: TokenCount | None = None
        self._last_api_tokens: TokenCount | None = None
        self._session_start_time = time.time()
        self._latest_budget_snapshot: dict[str, Any] | None = None
        self._latest_budget_snapshot_ts: float = 0.0

    def count_prompt_tokens(self, messages: list[BaseMessage]) -> TokenCount:
        """Count tokens in user prompt context"""
        token_count = self.tiktoken_counter.count_tokens(messages)
        self._last_prompt_tokens = token_count
        return token_count

    def track_llm_response(
        self, response: Any, messages: list[BaseMessage]
    ) -> TokenCount:
        """Track LLM response tokens with API fallback"""
        print("🔍 DEBUG: track_llm_response response type:", type(response))
        # Try to extract API tokens from response
        api_tokens = self._extract_api_tokens(response)
        print(f"🔍 DEBUG: Extracted API tokens: {api_tokens}")

        # Get token count (API or estimated)
        if api_tokens:
            # Use API tokens directly
            token_count = TokenCount(
                api_tokens[0],  # input_tokens
                api_tokens[1],  # output_tokens
                api_tokens[2],  # total_tokens
                is_estimated=False,  # Not estimated
                source="api",
            )
            print(f"🔍 DEBUG: Using API tokens: {token_count}")
        else:
            # Fallback to tiktoken estimation - count only the current request
            print("🔍 DEBUG: No API tokens, using tiktoken fallback")
            print("🔍 DEBUG: Counting tokens for current request only")

            # Extract only the current request to avoid counting duplicated context
            current_request = self._extract_current_request(messages)
            print(
                "🔍 DEBUG: Current request has %d messages (vs %d total)",
                len(current_request),
                len(messages),
            )

            token_count = self.tiktoken_counter.count_tokens(current_request)
            # Mark as estimated
            token_count = TokenCount(
                token_count.input_tokens,
                token_count.output_tokens,
                token_count.total_tokens,
                is_estimated=True,  # Mark as estimated
                source="tiktoken_estimation",
            )

        print(f"🔍 DEBUG: Token count result: {token_count}")
        self._last_api_tokens = token_count

        # Update cumulative tracking
        self._update_cumulative_tokens(token_count)
        print(f"🔍 DEBUG: Updated cumulative stats: {self.get_cumulative_stats()}")

        return token_count

    def _extract_current_request(
        self, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Extract only the current request (system + last user message)."""
        # Find the last HumanMessage (current user input)
        current_request = []
        last_user_message = None

        # Find the last user message
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                last_user_message = message
                break

        if last_user_message:
            # Collect system messages and the last user message
            system_messages = [
                msg for msg in messages if isinstance(msg, SystemMessage)
            ]
            current_request = [*system_messages, last_user_message]
        else:
            # Fallback to last message if no user message found
            current_request = messages[-1:] if messages else []

        return current_request

    def _extract_api_tokens(self, response: Any) -> tuple[int, int, int] | None:
        """Extract token usage from API response"""
        try:
            print(f"🔍 DEBUG: Response attributes: {dir(response)}")
            print(
                "🔍 DEBUG: Response has usage_metadata:",
                hasattr(response, "usage_metadata"),
            )
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                print(f"🔍 DEBUG: Usage metadata: {usage}")
                print(f"🔍 DEBUG: Usage metadata type: {type(usage)}")

                # Handle both dict and object formats
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    total_tokens = usage.get(
                        "total_tokens", input_tokens + output_tokens
                    )
                else:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    total_tokens = getattr(
                        usage, "total_tokens", input_tokens + output_tokens
                    )

                print(
                    "🔍 DEBUG: Extracted tokens: input=%s, output=%s, total=%s",
                    input_tokens,
                    output_tokens,
                    total_tokens,
                )
                return (input_tokens, output_tokens, total_tokens)
            print(
                "🔍 DEBUG: No usage_metadata found, checking for other token fields"
            )
            # Check for other common token usage fields
            if hasattr(response, "usage"):
                usage = response.usage
                print(f"🔍 DEBUG: Found usage field: {usage}")
            if hasattr(response, "token_usage"):
                usage = response.token_usage
                print(f"🔍 DEBUG: Found token_usage field: {usage}")
        except Exception as e:
            print(f"🔍 DEBUG: Error extracting API tokens: {e}")
        return None

    def _update_cumulative_tokens(self, token_count: TokenCount) -> None:
        """Update cumulative token tracking"""
        self.conversation_tokens += token_count.total_tokens
        self.session_tokens += token_count.total_tokens
        self.last_conversation_tokens = self.session_tokens
        self.message_count += 1

    def get_cumulative_stats(self) -> dict[str, Any]:
        """Get cumulative conversation statistics"""
        avg_tokens = (
            (self.conversation_tokens / self.message_count)
            if self.message_count > 0
            else 0
        )

        return {
            "conversation_tokens": self.conversation_tokens,
            "session_tokens": self.last_conversation_tokens,
            "message_count": self.message_count,
            "avg_tokens_per_message": int(avg_tokens)
        }

    def reset_session(self) -> None:
        """Reset session tokens while keeping conversation total"""
        self.session_tokens = 0
        self.last_conversation_tokens = 0

    def start_new_conversation(self) -> None:
        """Start tracking a new conversation"""
        self.session_tokens = 0
        self.last_conversation_tokens = 0

    def reset_current_conversation_budget(self) -> None:
        """Reset current conversation token budget for model switching"""
        # Reset the last API tokens to start fresh with new model
        self._last_api_tokens = None
        # Reset last prompt tokens as well
        self._last_prompt_tokens = None
        # Do not reset current conversation tokens on model switch; preserve continuity

    def get_last_prompt_tokens(self) -> TokenCount | None:
        """Get the last prompt token count"""
        return self._last_prompt_tokens

    def get_last_api_tokens(self) -> TokenCount | None:
        """Get the last API token count"""
        return self._last_api_tokens

    def get_token_display_info(self) -> dict[str, Any]:
        """Get formatted token display information"""
        return {
            "prompt_tokens": self._last_prompt_tokens,
            "api_tokens": self._last_api_tokens,
            "cumulative_stats": self.get_cumulative_stats()
        }

    def get_token_budget_info(self, context_window: int) -> dict[str, Any]:
        """Get token budget information for context window percentage calculation"""
        if not context_window or context_window <= 0:
            return {
                "used_tokens": 0,
                "context_window": 0,
                "percentage": 0.0,
                "remaining_tokens": 0,
                "status": TOKEN_STATUS_UNKNOWN,
            }

        # Prefer provider-reported token usage when available (ground truth),
        # fall back to the latest computed budget snapshot (budget moments),
        # then to any last-known API tokens.
        current_tokens = 0
        if self._last_api_tokens:
            try:
                # If provider returned a real usage figure, honor it.
                if (
                    not bool(getattr(self._last_api_tokens, "is_estimated", False))
                    and getattr(self._last_api_tokens, "source", "") == "api"
                ):
                    current_tokens = int(
                        getattr(self._last_api_tokens, "total_tokens", 0) or 0
                    )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed provider-usage preference in token budget: %s", exc
                )
        if (
            current_tokens <= 0
            and self._latest_budget_snapshot
            and isinstance(self._latest_budget_snapshot, dict)
        ):
            try:
                current_tokens = int(
                    self._latest_budget_snapshot.get("total_tokens", 0) or 0
                )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to read total_tokens from budget snapshot: %s", exc
                )
                current_tokens = 0
        if current_tokens <= 0 and self._last_api_tokens:
            current_tokens = int(getattr(self._last_api_tokens, "total_tokens", 0) or 0)

        percentage = (current_tokens / context_window) * 100.0
        remaining_tokens = max(0, context_window - current_tokens)

        # Determine status based on usage
        if percentage >= TOKEN_STATUS_CRITICAL_THRESHOLD:
            status = TOKEN_STATUS_CRITICAL
        elif percentage >= TOKEN_STATUS_WARNING_THRESHOLD:
            status = TOKEN_STATUS_WARNING
        elif percentage >= TOKEN_STATUS_MODERATE_THRESHOLD:
            status = TOKEN_STATUS_MODERATE
        else:
            status = TOKEN_STATUS_GOOD

        return {
            "used_tokens": current_tokens,
            "context_window": context_window,
            "percentage": round(percentage, 1),
            "remaining_tokens": remaining_tokens,
            "status": status
        }

    def set_budget_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Store latest budget snapshot for UI/stats consumption."""
        if not isinstance(snapshot, dict):
            return
        self._latest_budget_snapshot = snapshot
        self._latest_budget_snapshot_ts = float(snapshot.get("ts", time.time()))

    def get_budget_snapshot(self) -> dict[str, Any] | None:
        """Get the last stored budget snapshot (if any)."""
        return self._latest_budget_snapshot

    def refresh_budget_snapshot(
        self,
        *,
        agent: Any,
        conversation_id: str = "default",
        messages_override: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compute and store a fresh budget snapshot."""
        snap = compute_token_budget_snapshot(
            agent=agent,
            conversation_id=conversation_id,
            messages_override=messages_override,
        )
        self.set_budget_snapshot(snap)
        return snap


class UsageMetadataCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for API token tracking"""

    def __init__(self, token_tracker: ConversationTokenTracker) -> None:
        super().__init__()
        self.token_tracker = token_tracker

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Capture token usage when LLM call completes"""
        _ = kwargs
        try:
            if hasattr(response, "generations") and response.generations:
                generation = response.generations[0][0]
                if hasattr(generation, "message"):
                    self.token_tracker.track_llm_response(
                        response, [generation.message]
                    )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to capture usage metadata: %s", exc
            )


# Session-specific token tracker instances
_token_trackers: dict[str, ConversationTokenTracker] = {}

def get_token_tracker(session_id: str = "default") -> ConversationTokenTracker:
    """Get session-specific token tracker instance"""
    if session_id not in _token_trackers:
        _token_trackers[session_id] = ConversationTokenTracker()
    return _token_trackers[session_id]



def convert_chat_history_to_messages(
    history: list[dict[str, str]], current_message: str | None = None
) -> list[BaseMessage]:
    """
    Convert chat history format to BaseMessage format for token counting.

    Args:
        history: List of chat messages in format
            `[{"role": "user/assistant", "content": "..."}]`
        current_message: Optional current message to append

    Returns:
        List of BaseMessage objects
    """
    messages = []

    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(HumanMessage(content=content))  # For counting only
        elif role == "system":
            messages.append(SystemMessage(content=content))

    if current_message:
        messages.append(HumanMessage(content=current_message))

    return messages

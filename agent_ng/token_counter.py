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
    cost: float = 0.0  # Cost in USD credits

    @property
    def formatted(self) -> str:
        """Format token count for display"""
        if self.is_estimated:
            return f"~{self.total_tokens:,} total (estimated via {self.source})"
        cost_str = f" | ${self.cost:.4f}" if self.cost > 0 else ""
        return (
            f"{self.total_tokens:,} total "
            f"({self.input_tokens:,} input + {self.output_tokens:,} output)"
            f"{cost_str}"
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
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.debug("Failed to get encoding from cache: %s", exc)
            try:
                return tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            except Exception as fallback_exc:
                logger.debug("Failed to get encoding fallback: %s", fallback_exc)
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
        self, input_tokens: int, output_tokens: int, total_tokens: int, cost: float = 0.0
    ) -> None:
        """Set token count from API response"""
        self._last_api_tokens = TokenCount(
            input_tokens,
            output_tokens,
            total_tokens,
            is_estimated=False,
            source="api",
            cost=cost,
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
        # API-only totals (for "Vsego" (Total) and "Dialog" (Conversation)
        # display - only API counts, with estimates only for interrupted turns
        # without API).
        self.conversation_tokens = 0  # Total across all conversations (API-only)
        # Session-bound tokens (what UI calls "Диалог") (API-only)
        self.session_tokens = 0
        # Backward-compatible alias for older naming
        self.last_conversation_tokens = self.session_tokens
        self.message_count = 0
        # API-only totals for accurate average calculation (excludes estimates except
        # for interrupted turns without API counts).
        self._api_only_tokens = 0
        self._api_only_message_count = 0
        # Track if the last finalized turn used an estimate (interrupted without API).
        self._last_turn_used_estimate = False
        self._last_prompt_tokens: TokenCount | None = None
        self._last_api_tokens: TokenCount | None = None
        self._session_start_time = time.time()
        self._latest_budget_snapshot: dict[str, Any] | None = None
        self._latest_budget_snapshot_ts: float = 0.0
        # Per-QA-turn accumulated usage across multiple LLM calls (iterations).
        # This must not affect avg-per-message until the turn is finalized.
        self._turn_active: bool = False
        self._turn_input_tokens: int = 0
        self._turn_output_tokens: int = 0
        self._turn_total_tokens: int = 0
        self._turn_cost: float = 0.0  # Cost per turn in USD
        # Per-QA-turn monotonic estimate (used only when API usage isn't available and
        # turn was interrupted/failed).
        self._turn_estimated_total_tokens: int = 0
        # Last finalized turn estimate (for display after turn completes).
        self._last_turn_estimated_total_tokens: int = 0
        # Track if the turn was interrupted/failed (so we know to use estimates).
        self._turn_interrupted: bool = False
        # Cost tracking: per-conversation and overall totals
        self.conversation_cost: float = 0.0  # Total cost across all conversations
        self.session_cost: float = 0.0  # Cost for current conversation/session
        self._last_turn_cost: float = 0.0  # Cost of last finalized turn

    def count_prompt_tokens(self, messages: list[BaseMessage]) -> TokenCount:
        """Count tokens in user prompt context"""
        token_count = self.tiktoken_counter.count_tokens(messages)
        self._last_prompt_tokens = token_count
        return token_count

    def track_llm_response(
        self, response: Any, messages: list[BaseMessage]
    ) -> TokenCount:
        """Track LLM response tokens with API fallback"""
        logger.debug("track_llm_response response type: %s", type(response))
        # Try to extract API tokens from response
        api_tokens = self._extract_api_tokens(response)
        logger.debug("Extracted API tokens: %s", api_tokens)

        # Get token count (API or estimated)
        if api_tokens:
            # Use API tokens directly
            cost = api_tokens[3] if len(api_tokens) >= 4 else 0.0
            token_count = TokenCount(
                api_tokens[0],  # input_tokens
                api_tokens[1],  # output_tokens
                api_tokens[2],  # total_tokens
                is_estimated=False,  # Not estimated
                source="api",
                cost=cost,
            )
            logger.debug("Using API tokens: %d", token_count)
        else:
            # Fallback to tiktoken estimation - count only the current request
            logger.debug("No API tokens, using tiktoken fallback")
            logger.debug("Counting tokens for current request only")

            # Extract only the current request to avoid counting duplicated context
            current_request = self._extract_current_request(messages)
            logger.debug(
                "Current request has %d messages (vs %d total)",
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

        logger.debug("Token count result: %d", token_count)
        self._last_api_tokens = token_count

        # Update cumulative tracking
        self._update_cumulative_tokens(token_count)
        logger.debug("Updated cumulative stats: %s", self.get_cumulative_stats())

        return token_count

    def begin_turn(self) -> None:
        """Begin a new QA turn usage accumulation (may include multiple LLM calls)."""
        self._turn_active = True
        self._turn_input_tokens = 0
        self._turn_output_tokens = 0
        self._turn_total_tokens = 0
        self._turn_cost = 0.0
        self._turn_estimated_total_tokens = 0
        self._turn_interrupted = False

    def _update_turn_estimate(self, estimated_total_tokens: int) -> None:
        """Update monotonic per-turn estimate (best-effort)."""
        try:
            val = int(estimated_total_tokens or 0)
        except Exception:
            return
        self._turn_estimated_total_tokens = max(self._turn_estimated_total_tokens, val)

    def update_turn_usage_from_api(self, response: Any) -> bool:
        """Update turn usage with provider-reported accumulated usage."""
        if not self._turn_active:
            # Be resilient: allow accumulation even if begin_turn wasn't called.
            self.begin_turn()

        api_tokens = self._extract_api_tokens(response)
        if not api_tokens:
            return False

        input_tokens, output_tokens, total_tokens = api_tokens[0], api_tokens[1], api_tokens[2]
        cost = api_tokens[3] if len(api_tokens) >= 4 else 0.0
        try:
            # API returns accumulated spending per turn - use replacement, not addition
            self._turn_input_tokens = int(input_tokens or 0)
            self._turn_output_tokens = int(output_tokens or 0)
            self._turn_total_tokens = int(total_tokens or 0)
            self._turn_cost = float(cost or 0.0)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to accumulate LLM call usage: %s", exc
            )
            return False

        # Expose current per-turn accumulated usage for UI during iterations.
        self._last_api_tokens = TokenCount(
            self._turn_input_tokens,
            self._turn_output_tokens,
            self._turn_total_tokens,
            is_estimated=False,
            source="api",
            cost=self._turn_cost,
        )
        return True

    def get_turn_estimated_total_tokens(self) -> int:
        """Get the current per-turn monotonic estimate."""
        return int(self._turn_estimated_total_tokens or 0)

    def get_message_display_total_tokens(self) -> int:
        """Tokens to display in 'Сообщение' right now.

        Shows only API-provided true counts, falling back to estimates only when
        the turn was interrupted/failed before API counts were received.

        Priority:
        - If we have accumulated API usage for this turn: use it (always preferred).
        - Else if turn is active: return 0 (wait for API counts, don't show estimates).
        - Else if turn was interrupted/failed: use estimate as fallback.
        - Else (turn completed successfully): return 0 (should have API counts).
        """
        # Always prefer accumulated API usage if available.
        if int(self._turn_total_tokens or 0) > 0:
            return int(self._turn_total_tokens)

        # During active turn: don't show estimates, wait for API counts.
        if self._turn_active:
            return 0

        # Turn is not active: only use estimate if it was interrupted/failed.
        if self._turn_interrupted and int(self._turn_estimated_total_tokens or 0) > 0:
            return int(self._turn_estimated_total_tokens)

        # Turn completed successfully but no API counts? This shouldn't happen,
        # but return 0 to avoid showing stale estimates.
        return 0

    def finalize_turn_usage(
        self, response: Any | None, messages: list[BaseMessage] | None = None
    ) -> TokenCount | None:
        """Finalize the current QA turn.

        - If we successfully accumulated per-iteration API usage, commit it once to
          totals and increment message_count once (preserves avg-per-message).
        - If response is None, mark turn as interrupted and use estimate if no API
          counts.
        - Otherwise, fall back to the legacy `track_llm_response` behavior.
        """
        # Mark as interrupted if response is None (user stopped the turn).
        if response is None:
            self._turn_interrupted = True

        if self._turn_total_tokens > 0:
            # We have API counts: use them (successful completion or interrupted
            # with API).
            token_count = TokenCount(
                self._turn_input_tokens,
                self._turn_output_tokens,
                self._turn_total_tokens,
                is_estimated=False,
                source="api",
                cost=self._turn_cost,
            )
            self._last_api_tokens = token_count
            self._last_turn_cost = self._turn_cost
            # For normal completions with API values, don't contaminate with estimates
            self._last_turn_estimated_total_tokens = 0
            # Reset turn estimates since the turn is complete
            self._turn_estimated_total_tokens = 0

            # Commit to totals once per QA turn.
            self.conversation_tokens += token_count.total_tokens
            self.session_tokens += token_count.total_tokens
            self.last_conversation_tokens = self.session_tokens
            self.message_count += 1
            # Track costs
            self.conversation_cost += self._turn_cost
            self.session_cost += self._turn_cost

            # Track API-only totals for accurate average calculation.
            self._api_only_tokens += token_count.total_tokens
            self._api_only_message_count += 1
            self._last_turn_used_estimate = False

            self._turn_active = False
            return token_count

        # No API totals: only commit estimate if turn was interrupted/failed.
        if (
            self._turn_interrupted
            and int(self._turn_estimated_total_tokens or 0) > 0
        ):
            est = int(self._turn_estimated_total_tokens)
            token_count = TokenCount(
                est,
                0,
                est,
                is_estimated=True,
                source="estimate",
            )
            self._last_api_tokens = token_count
            self._last_turn_estimated_total_tokens = est

            # Commit to overall totals (for "Vsego" and "Dialog" display).
            self.conversation_tokens += token_count.total_tokens
            self.session_tokens += token_count.total_tokens
            self.last_conversation_tokens = self.session_tokens
            self.message_count += 1

            # For average calculation: include estimate only for interrupted turn
            # (this is the exception case where we use estimate in average).
            self._api_only_tokens += token_count.total_tokens
            self._api_only_message_count += 1
            self._last_turn_used_estimate = True

            self._turn_active = False
            return token_count

        # Fallback: preserve legacy behavior (includes message_count increment).
        # This handles cases where finalize_turn_usage is called with a response
        # but we haven't accumulated API counts (shouldn't happen normally).
        self._turn_active = False
        if response is None or not messages:
            return None
        return self.track_llm_response(response, messages)

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

    def _extract_api_tokens(self, response: Any) -> tuple[int, int, int, float] | None:
        """Extract token usage and cost from API response

        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens, cost) or None
        """
        try:
            logger = logging.getLogger(__name__)
            logger.debug("Extracting API tokens from response type=%s", type(response))
            cost = 0.0

            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                logger.debug("Found usage_metadata type=%s", type(usage))

                # Handle both dict and object formats
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    total_tokens = usage.get(
                        "total_tokens", input_tokens + output_tokens
                    )
                    # Extract cost if available (OpenRouter uses "cost" field)
                    cost = float(usage.get("cost", 0.0) or 0.0)
                else:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    total_tokens = getattr(
                        usage, "total_tokens", input_tokens + output_tokens
                    )
                    # Extract cost if available (OpenRouter uses "cost" field)
                    cost = float(getattr(usage, "cost", 0.0) or 0.0)

                logger.debug(
                    "Extracted tokens from usage_metadata: input=%s output=%s total=%s cost=%s",
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost,
                )
                return (input_tokens, output_tokens, total_tokens, cost)

            # Provider/model wrappers often attach usage to response_metadata or
            # additional_kwargs.
            candidate_dicts: list[dict[str, Any]] = []
            response_meta = getattr(response, "response_metadata", None)
            if isinstance(response_meta, dict):
                candidate_dicts.append(response_meta)
            additional = getattr(response, "additional_kwargs", None)
            if isinstance(additional, dict):
                candidate_dicts.append(additional)

            # Also consider direct `usage` / `token_usage` fields when they are dicts.
            usage_field = getattr(response, "usage", None)
            if isinstance(usage_field, dict):
                candidate_dicts.append({"usage": usage_field})
            token_usage_field = getattr(response, "token_usage", None)
            if isinstance(token_usage_field, dict):
                candidate_dicts.append({"usage": token_usage_field})

            for d in candidate_dicts:
                usage = (
                    d.get("usage_metadata")
                    or d.get("usage")
                    or d.get("token_usage")
                    or d.get("usage_details")
                )
                if not isinstance(usage, dict):
                    continue

                # Normalize common shapes:
                # OpenAI/OpenRouter {prompt_tokens, completion_tokens, total_tokens} and
                # LangChain usage_metadata {input_tokens, output_tokens, total_tokens}.
                if "input_tokens" in usage:
                    input_tokens = usage.get("input_tokens", 0)
                else:
                    input_tokens = usage.get("prompt_tokens", 0)

                if "output_tokens" in usage:
                    output_tokens = usage.get("output_tokens", 0)
                else:
                    output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", None)
                if total_tokens is None:
                    try:
                        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
                    except Exception as calc_exc:
                        logging.getLogger(__name__).debug(
                            "Failed to calculate total tokens: %s", calc_exc
                        )
                        total_tokens = 0

                # Extract cost if available (OpenRouter uses "cost" field)
                cost = float(usage.get("cost", 0.0) or 0.0)

                if int(total_tokens or 0) > 0:
                    logger.debug(
                        "Extracted tokens from metadata dict: input=%s output=%s total=%s cost=%s",  # noqa: E501
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        cost,
                    )
                    return (
                        int(input_tokens or 0),
                        int(output_tokens or 0),
                        int(total_tokens or 0),
                        cost,
                    )

        except Exception as e:
            logging.getLogger(__name__).debug("Error extracting API tokens: %s", e)
        return None

    def _extract_api_cost(self, response: Any) -> float:
        """Extract cost from API response (helper method for backward compatibility)"""
        result = self._extract_api_tokens(response)
        if result and len(result) >= 4:
            return result[3]  # cost is the 4th element
        return 0.0

    def _update_cumulative_tokens(self, token_count: TokenCount) -> None:
        """Update cumulative token tracking.

        Only updates totals with API counts. Estimates are excluded (legacy path
        doesn't know if turn was interrupted, so we exclude to avoid doubling).
        """
        # Legacy method: only update totals with API counts to avoid doubling.
        # Estimates are excluded unless we know the turn was interrupted (which
        # is handled in finalize_turn_usage).
        if not token_count.is_estimated:
            self.conversation_tokens += token_count.total_tokens
            self.session_tokens += token_count.total_tokens
            self.last_conversation_tokens = self.session_tokens
            self.message_count += 1
            # Track costs
            self.conversation_cost += token_count.cost
            self.session_cost += token_count.cost
            self._last_turn_cost = token_count.cost

            # Track API-only totals for accurate average calculation.
            self._api_only_tokens += token_count.total_tokens
            self._api_only_message_count += 1
            self._last_turn_used_estimate = False
        # else: Legacy estimate excluded to avoid doubling (new path handles
        # interrupted turns correctly via finalize_turn_usage).

    def get_cumulative_stats(self) -> dict[str, Any]:
        """Get cumulative conversation statistics.

        All metrics are API-based:
        - conversation_tokens (Total): API counts only, with estimates
          only for interrupted turns without API counts.
        - session_tokens (Conversation): API counts only, with
          estimates only for interrupted turns without API counts.
        - avg_tokens_per_message (Average per
          message): Per-conversation average (session_tokens / message_count).
        - Cost tracking: per-turn, per-conversation, and overall totals.
        """
        # Calculate average per message for current conversation only
        # Use session_tokens (per-conversation) divided by message_count (per-conversation)
        avg_tokens = (
            (self.last_conversation_tokens / self.message_count)
            if self.message_count > 0
            else 0
        )

        # Get last turn token counts if available
        last_input_tokens = 0
        last_output_tokens = 0
        if self._last_api_tokens:
            last_input_tokens = self._last_api_tokens.input_tokens
            last_output_tokens = self._last_api_tokens.output_tokens

        # Get cumulative input/output token counts
        # For now, we track per-turn but need to accumulate for totals
        # This is a simplified version - in practice, we'd need to track these separately
        # For now, we'll use the last turn's breakdown as an indicator

        # conversation_tokens and session_tokens are already API-only (only updated
        # with API counts or interrupted-turn estimates in finalize_turn_usage).
        return {
            "conversation_tokens": self.conversation_tokens,
            "session_tokens": self.last_conversation_tokens,
            "message_count": self.message_count,
            "avg_tokens_per_message": int(avg_tokens),
            # Cost tracking
            "turn_cost": self._last_turn_cost,
            "conversation_cost": self.session_cost,
            "total_cost": self.conversation_cost,
            # Token breakdowns
            "last_input_tokens": last_input_tokens,
            "last_output_tokens": last_output_tokens,
            # Per-turn breakdown (for current/last turn)
            "turn_input_tokens": self._turn_input_tokens if not self._turn_active else 0,
            "turn_output_tokens": self._turn_output_tokens if not self._turn_active else 0,
        }

    def reset_session(self) -> None:
        """Reset session tokens while keeping conversation total"""
        self.session_tokens = 0
        self.last_conversation_tokens = 0
        self.session_cost = 0.0
        # Reset API-only tracking for session (but keep conversation total).
        # Note: We keep _api_only_tokens and _api_only_message_count for
        # accurate average across all conversations.

    def start_new_conversation(self) -> None:
        """Start tracking a new conversation"""
        self.session_tokens = 0
        self.last_conversation_tokens = 0
        self.session_cost = 0.0
        # Reset API-only tracking for new conversation (but keep conversation total).
        # Note: We keep _api_only_tokens and _api_only_message_count for
        # accurate average across all conversations.
        # Reset per-conversation display stats
        self._latest_budget_snapshot = None
        self._latest_budget_snapshot_ts = 0.0
        self._turn_estimated_total_tokens = 0
        self._turn_total_tokens = 0
        self._turn_input_tokens = 0
        self._turn_output_tokens = 0
        self._turn_cost = 0.0
        self._turn_active = False
        self._turn_interrupted = False
        self._last_api_tokens = None
        self._last_turn_estimated_total_tokens = 0
        self._last_turn_cost = 0.0
        self._last_turn_used_estimate = False
        # Reset per-conversation message count for average calculation
        # (but keep cumulative _api_only_message_count for cross-conversation average)
        self.message_count = 0

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

        # For the UI "Сообщение" metric we want a monotonic, per-turn total:
        # - sum of API usage across iterations when available
        # - otherwise, a monotonic estimate (snapshot-based) suitable for interruption
        #   fallback
        current_tokens = self.get_message_display_total_tokens()

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
        # Feed per-turn estimate from snapshot totals during iterations (monotonic).
        if self._turn_active:
            try:
                self._update_turn_estimate(int(snapshot.get("total_tokens", 0) or 0))
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to update per-turn estimate from snapshot: %s", exc
                )

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
        # Include tool schemas (actually sent to LLM) but no other overhead
        snap = compute_token_budget_snapshot(
            agent=agent,
            conversation_id=conversation_id,
            messages_override=messages_override,
            include_overhead=True,   # Include tool schemas only
            add_json_overhead=False, # No JSON overhead inflation
        )
        self.set_budget_snapshot(snap)
        # Debug logging for estimated vs API comparison
        logger = logging.getLogger(__name__)
        logger.debug("Budget snapshot computed: %s total tokens", snap["total_tokens"])
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

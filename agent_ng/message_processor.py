"""
Message Processor Module
========================

This module handles all message formatting, processing, and response extraction functionality.
It provides utilities for preparing messages for LLMs and extracting answers from responses.

Key Features:
- Message formatting and preparation
- Chat history management
- Response extraction and parsing
- Tool call detection and processing
- Message component debugging

Usage:
    from message_processor import MessageProcessor

    processor = MessageProcessor()
    messages = processor.format_messages(question, chat_history)
    answer = processor.extract_final_answer(response)
"""

import json
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage


@dataclass
class MessageContext:
    """Represents the context for message processing"""
    question: str
    reference: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    use_tools: bool = True


class MessageProcessor:
    """Handles message formatting, processing, and response extraction"""

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.sys_msg = SystemMessage(content=self.system_prompt)

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt"""
        return """You are a helpful AI assistant. You can use tools to help answer questions. 
When you have enough information to provide a complete answer, use the submit_answer tool with your final response."""

    def format_messages(self, context: MessageContext) -> List[BaseMessage]:
        """
        Format the message list for the LLM, including system prompt, optional prior chat history,
        question, and optional reference answer.

        Args:
            context: MessageContext containing question, reference, chat_history, etc.

        Returns:
            List[BaseMessage]: List of message objects for the LLM.
        """
        messages = [self.sys_msg]

        # Append prior chat history with full tool execution context
        if context.chat_history:
            for turn in context.chat_history:
                # Add human message
                if turn.get('role') == 'user' and turn.get('content'):
                    messages.append(HumanMessage(content=turn['content']))

                # Add AI message with tool calls if present
                if turn.get('role') == 'assistant':
                    ai_content = turn.get('content', '')
                    tool_calls = turn.get('tool_calls', [])

                    if tool_calls:
                        # Create AI message with tool calls
                        ai_message = AIMessage(content=ai_content)
                        ai_message.tool_calls = tool_calls
                        messages.append(ai_message)
                    elif ai_content:
                        # Regular AI message
                        messages.append(AIMessage(content=ai_content))

                # Add tool results if present
                if turn.get('role') == 'tool' and turn.get('tool_call_id'):
                    tool_message = ToolMessage(
                        content=turn.get('content', ''),
                        tool_call_id=turn.get('tool_call_id'),
                        name=turn.get('tool_name', '')
                    )
                    messages.append(tool_message)

        # Add current question
        messages.append(HumanMessage(content=context.question))

        # Add reference answer if provided
        if context.reference:
            reference_msg = f"\n\nReference Answer: {context.reference}"
            messages.append(HumanMessage(content=reference_msg))

        return messages

    def format_messages_simple(self, question: str, reference: Optional[str] = None, 
                             chat_history: Optional[List[Dict[str, Any]]] = None) -> List[BaseMessage]:
        """
        Simple wrapper for format_messages with individual parameters.

        Args:
            question: The question to answer
            reference: Optional reference answer
            chat_history: Optional chat history

        Returns:
            List[BaseMessage]: Formatted messages
        """
        context = MessageContext(
            question=question,
            reference=reference,
            chat_history=chat_history,
            system_prompt=self.system_prompt
        )
        return self.format_messages(context)

    def extract_final_answer(self, response: Any) -> str:
        """
        Extract the final answer from the LLM response using structured output approach.
        This agent is tool-only and requires the submit_answer tool to be used.

        Args:
            response: The LLM response object.

        Returns:
            str: The extracted final answer string.
        """
        # Extract using structured output (tool-only approach)
        structured_answer = self._extract_structured_final_answer(response)
        if structured_answer:
            return structured_answer

        # Fallback to content extraction
        content_answer = self._extract_content_answer(response)
        if content_answer:
            return content_answer

        # Last resort: return the raw response
        return str(response)

    def _extract_structured_final_answer(self, response: Any) -> Optional[str]:
        """Extract answer from structured tool calls"""
        try:
            # Check for tool calls in the response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    # LangChain tool calls are dictionaries
                    if tool_call.get('name', '') == 'submit_answer':
                        args = tool_call.get('args', {})
                        return args.get('answer', '') if isinstance(args, dict) else ''

            # Check for function calls (legacy format)
            if hasattr(response, 'function_call') and response.function_call:
                if response.function_call.get('name') == 'submit_answer':
                    args = response.function_call.get('arguments', '{}')
                    try:
                        args_dict = json.loads(args)
                        return args_dict.get('answer', '')
                    except json.JSONDecodeError:
                        return args

            # Check for content with tool call patterns
            content = getattr(response, 'content', '')
            if isinstance(content, str):
                # Look for submit_answer tool call in content
                submit_pattern = r'submit_answer\([^)]*answer[^)]*["\']([^"\']+)["\']'
                match = re.search(submit_pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1)

                # Look for JSON tool call format
                json_pattern = r'{"name":\s*"submit_answer"[^}]*"answer":\s*"([^"]+)"'
                match = re.search(json_pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1)

        except Exception as e:
            print(f"[MessageProcessor] Error extracting structured answer: {e}")

        return None

    def _extract_content_answer(self, response: Any) -> Optional[str]:
        """Extract answer from response content"""
        try:
            content = getattr(response, 'content', '')
            if not content:
                return None

            if isinstance(content, str):
                # Look for answer patterns in content
                answer_patterns = [
                    r'Answer:\s*(.+?)(?:\n\n|\n$|$)',
                    r'Final Answer:\s*(.+?)(?:\n\n|\n$|$)',
                    r'Here is the answer:\s*(.+?)(?:\n\n|\n$|$)',
                    r'The answer is:\s*(.+?)(?:\n\n|\n$|$)',
                ]

                for pattern in answer_patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                        if answer and len(answer) > 10:  # Ensure it's a substantial answer
                            return answer

                # If no pattern matches, return the content as-is if it looks like an answer
                if len(content.strip()) > 10 and not content.strip().startswith('I need to'):
                    return content.strip()

        except Exception as e:
            print(f"[MessageProcessor] Error extracting content answer: {e}")

        return None

    def force_final_answer(self, messages: List[BaseMessage], tool_results_history: List[Any], 
                          llm: Any) -> Any:
        """
        Force the LLM to provide a final answer by adding a reminder prompt.
        Tool results are already available in the message history as ToolMessage objects.

        Args:
            messages: Current message list (contains tool results as ToolMessage objects)
            tool_results_history: History of tool results (for reference, not used in prompt)
            llm: LLM instance

        Returns:
            Response from LLM with structured answer via submit_answer tool
        """
        # Create a reminder message to force final answer
        reminder_msg = HumanMessage(content="""
Please provide your final answer now. You have all the information you need from the tool results above. 
Use the submit_answer tool to provide your complete response.
""")

        # Add reminder to messages
        messages_with_reminder = messages + [reminder_msg]

        try:
            # Make LLM call with reminder
            response = llm.invoke(messages_with_reminder)
            return response
        except Exception as e:
            print(f"[MessageProcessor] Error in force_final_answer: {e}")
            # Fallback: return a simple text response
            return AIMessage(content="I apologize, but I encountered an error while trying to provide a final answer. Please try again.")

    def print_message_components(self, msg: Any, msg_index: int = 0):
        """
        Smart, agnostic message component printer that dynamically discovers and prints all relevant attributes.
        Uses introspection, JSON-like handling, and smart filtering for optimal output.
        """
        separator = "------------------------------------------------\n"
        print(separator) 
        print(f"Message {msg_index}:")

        # Get message type dynamically
        msg_type = getattr(msg, 'type', 'unknown')
        print(f"  type: {msg_type}")

        # Define priority attributes to check first (most important)
        priority_attrs = ['content', 'tool_calls', 'function_call', 'name', 'tool_call_id']

        # Print priority attributes first
        for attr in priority_attrs:
            if hasattr(msg, attr):
                value = getattr(msg, attr)
                if value is not None:
                    self._print_attribute(attr, value)

        # Print other attributes (excluding private ones and already printed ones)
        for attr_name in dir(msg):
            if (not attr_name.startswith('_') and 
                attr_name not in priority_attrs and 
                not callable(getattr(msg, attr_name))):
                try:
                    value = getattr(msg, attr_name)
                    if value is not None and value != '':
                        self._print_attribute(attr_name, value)
                except Exception:
                    pass  # Skip attributes that can't be accessed

        print(separator)

    def _print_attribute(self, attr_name: str, value: Any):
        """Print an attribute value in a formatted way"""
        if isinstance(value, (str, int, float, bool)):
            print(f"  {attr_name}: {value}")
        elif isinstance(value, list):
            if len(value) == 0:
                print(f"  {attr_name}: []")
            else:
                print(f"  {attr_name}: [list with {len(value)} items]")
                for i, item in enumerate(value[:3]):  # Show first 3 items
                    print(f"    [{i}]: {str(item)[:100]}...")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more items")
        elif isinstance(value, dict):
            if len(value) == 0:
                print(f"  {attr_name}: {{}}")
            else:
                print(f"  {attr_name}: {{dict with {len(value)} keys}}")
                for key, val in list(value.items())[:3]:  # Show first 3 key-value pairs
                    print(f"    {key}: {str(val)[:100]}...")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more keys")
        else:
            print(f"  {attr_name}: {type(value).__name__} - {str(value)[:100]}...")

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract tool calls from LLM response.

        Args:
            response: LLM response object

        Returns:
            List[Dict]: List of tool call dictionaries
        """
        tool_calls = []

        try:
            # Check for tool_calls attribute
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls.extend(response.tool_calls)

            # Check for function_call attribute (legacy)
            if hasattr(response, 'function_call') and response.function_call:
                tool_calls.append(response.function_call)

            # Check content for tool call patterns
            content = getattr(response, 'content', '')
            if isinstance(content, str):
                # Look for JSON tool call patterns
                json_pattern = r'{"name":\s*"[^"]+",\s*"args":\s*{[^}]+}}'
                matches = re.findall(json_pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        tool_call = json.loads(match)
                        tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"[MessageProcessor] Error extracting tool calls: {e}")

        return tool_calls

    def has_tool_calls(self, response: Any) -> bool:
        """Check if response contains tool calls"""
        return len(self.extract_tool_calls(response)) > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get message processor statistics"""
        return {
            "system_prompt_length": len(self.system_prompt),
            "has_system_message": self.sys_msg is not None
        }


# Global message processor instance
_message_processor = None

def get_message_processor(system_prompt: str = None) -> MessageProcessor:
    """Get the global message processor instance"""
    global _message_processor
    if _message_processor is None or (system_prompt and system_prompt != _message_processor.system_prompt):
        _message_processor = MessageProcessor(system_prompt)
    return _message_processor

def reset_message_processor():
    """Reset the global message processor instance"""
    global _message_processor
    _message_processor = None

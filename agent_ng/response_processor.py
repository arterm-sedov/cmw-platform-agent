"""
Response Processor Module
========================

This module handles all response processing, extraction, and formatting functionality.
It provides utilities for extracting answers from LLM responses and processing various response formats.

Key Features:
- Response extraction and parsing
- Structured answer extraction
- Content-based answer extraction
- Response validation and formatting
- Debug output and logging

Usage:
    from response_processor import ResponseProcessor

    processor = ResponseProcessor()
    answer = processor.extract_final_answer(response)
    formatted = processor.format_response(response)
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage


@dataclass
class ProcessedResponse:
    """Represents a processed response"""
    content: str
    tool_calls: List[Dict[str, Any]]
    has_tool_calls: bool
    is_final_answer: bool
    confidence: float
    metadata: Dict[str, Any]


class ResponseProcessor:
    """Handles response processing, extraction, and formatting"""

    def __init__(self):
        self.answer_patterns = [
            r'Answer:\s*(.+?)(?:\n\n|\n$|$)',
            r'Final Answer:\s*(.+?)(?:\n\n|\n$|$)',
            r'Here is the answer:\s*(.+?)(?:\n\n|\n$|$)',
            r'The answer is:\s*(.+?)(?:\n\n|\n$|$)',
            r'Based on the information:\s*(.+?)(?:\n\n|\n$|$)',
            r'In conclusion:\s*(.+?)(?:\n\n|\n$|$)',
        ]

        self.tool_call_patterns = [
            r'submit_answer\([^)]*answer[^)]*["\']([^"\']+)["\']',
            r'{"name":\s*"submit_answer"[^}]*"answer":\s*"([^"]+)"',
            r'<tool_call name="submit_answer">.*?<answer>(.*?)</answer>',
        ]

    def extract_final_answer(self, response: Any) -> str:
        """
        Extract the final answer from the LLM response using multiple strategies.

        Args:
            response: The LLM response object

        Returns:
            str: The extracted final answer string
        """
        # Try structured extraction first
        structured_answer = self.extract_structured_answer(response)
        if structured_answer:
            return structured_answer

        # Fallback to content extraction
        content_answer = self.extract_content_answer(response)
        if content_answer:
            return content_answer

        # Last resort: return the raw response
        return str(response)

    def extract_structured_answer(self, response: Any) -> Optional[str]:
        """
        Extract answer from structured tool calls or JSON format.

        Args:
            response: The LLM response object

        Returns:
            Optional[str]: Extracted answer if found
        """
        try:
            # Check for tool calls in the response
            tool_calls = self.extract_tool_calls(response)
            for tool_call in tool_calls:
                # Tool calls from extract_tool_calls are already dictionaries
                if tool_call.get('name') == 'submit_answer':
                    args = tool_call.get('args', {})
                    answer = args.get('answer', '') if isinstance(args, dict) else ''
                    if answer:
                        return answer

            # Check for function calls (legacy format)
            if hasattr(response, 'function_call') and response.function_call:
                if response.function_call.get('name') == 'submit_answer':
                    args = response.function_call.get('arguments', '{}')
                    try:
                        args_dict = json.loads(args)
                        answer = args_dict.get('answer', '')
                        if answer:
                            return answer
                    except json.JSONDecodeError:
                        pass

            # Check content for tool call patterns
            content = getattr(response, 'content', '')
            if isinstance(content, str):
                for pattern in self.tool_call_patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                        if answer:
                            return answer

        except Exception as e:
            print(f"[ResponseProcessor] Error extracting structured answer: {e}")

        return None

    def extract_content_answer(self, response: Any) -> Optional[str]:
        """
        Extract answer from response content using pattern matching.

        Args:
            response: The LLM response object

        Returns:
            Optional[str]: Extracted answer if found
        """
        try:
            content = getattr(response, 'content', '')
            if not content or not isinstance(content, str):
                return None

            # Try answer patterns
            for pattern in self.answer_patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    if answer and len(answer) > 10:  # Ensure it's a substantial answer
                        return answer

            # If no pattern matches, check if content looks like an answer
            content = content.strip()
            if (len(content) > 10 and 
                not content.startswith('I need to') and 
                not content.startswith('Let me') and
                not content.startswith('I will')):
                return content

        except Exception as e:
            print(f"[ResponseProcessor] Error extracting content answer: {e}")

        return None

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
            print(f"[ResponseProcessor] Error extracting tool calls: {e}")

        return tool_calls

    def has_tool_calls(self, response: Any) -> bool:
        """Check if response contains tool calls"""
        return len(self.extract_tool_calls(response)) > 0

    def is_final_answer(self, response: Any) -> bool:
        """Check if response contains a final answer"""
        return self.extract_final_answer(response) is not None

    def process_response(self, response: Any) -> ProcessedResponse:
        """
        Process a response and extract all relevant information.

        Args:
            response: LLM response object

        Returns:
            ProcessedResponse: Processed response with metadata
        """
        content = getattr(response, 'content', '')
        tool_calls = self.extract_tool_calls(response)
        has_tool_calls = len(tool_calls) > 0
        final_answer = self.extract_final_answer(response)
        is_final_answer = final_answer is not None

        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(response, final_answer)

        # Extract metadata
        metadata = {
            'response_type': type(response).__name__,
            'content_length': len(str(content)),
            'tool_call_count': len(tool_calls),
            'has_structured_output': has_tool_calls,
            'extraction_method': self._get_extraction_method(response, final_answer)
        }

        return ProcessedResponse(
            content=str(content),
            tool_calls=tool_calls,
            has_tool_calls=has_tool_calls,
            is_final_answer=is_final_answer,
            confidence=confidence,
            metadata=metadata
        )

    def _calculate_confidence(self, response: Any, final_answer: str) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.5  # Base confidence

        # Increase confidence for structured outputs
        if self.has_tool_calls(response):
            confidence += 0.3

        # Increase confidence for longer, more detailed answers
        if final_answer and len(final_answer) > 50:
            confidence += 0.2

        # Increase confidence for answers with specific patterns
        if final_answer and any(pattern in final_answer.lower() for pattern in 
                              ['based on', 'according to', 'the answer is', 'therefore']):
            confidence += 0.1

        return min(confidence, 1.0)

    def _get_extraction_method(self, response: Any, final_answer: str) -> str:
        """Get the method used to extract the final answer"""
        if not final_answer:
            return "none"

        if self.extract_structured_answer(response):
            return "structured"
        elif self.extract_content_answer(response):
            return "content_patterns"
        else:
            return "raw_content"

    def format_response(self, response: Any, include_metadata: bool = False) -> str:
        """
        Format a response for display.

        Args:
            response: LLM response object
            include_metadata: Whether to include metadata

        Returns:
            str: Formatted response string
        """
        processed = self.process_response(response)

        formatted = f"Content: {processed.content}\n"

        if processed.has_tool_calls:
            formatted += f"Tool Calls: {len(processed.tool_calls)}\n"
            for i, tool_call in enumerate(processed.tool_calls):
                formatted += f"  {i+1}. {tool_call.get('name', 'unknown')}\n"

        if include_metadata:
            formatted += f"Confidence: {processed.confidence:.2f}\n"
            formatted += f"Extraction Method: {processed.metadata['extraction_method']}\n"

        return formatted

    def validate_response(self, response: Any) -> Dict[str, Any]:
        """
        Validate a response and return validation results.

        Args:
            response: LLM response object

        Returns:
            Dict: Validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        try:
            # Check if response has content
            content = getattr(response, 'content', '')
            if not content:
                validation['is_valid'] = False
                validation['errors'].append("Response has no content")

            # Check content length
            if isinstance(content, str) and len(content) < 10:
                validation['warnings'].append("Response content is very short")

            # Check for tool calls if expected
            tool_calls = self.extract_tool_calls(response)
            if not tool_calls and not self.is_final_answer(response):
                validation['warnings'].append("Response has no tool calls or final answer")

            # Check for common issues
            if isinstance(content, str):
                if content.lower().startswith('i need to'):
                    validation['suggestions'].append("Response suggests incomplete processing")
                if 'error' in content.lower():
                    validation['warnings'].append("Response contains error mentions")

        except Exception as e:
            validation['is_valid'] = False
            validation['errors'].append(f"Validation error: {str(e)}")

        return validation


    def get_stats(self) -> Dict[str, Any]:
        """Get response processor statistics"""
        return {
            'answer_patterns_count': len(self.answer_patterns),
            'tool_call_patterns_count': len(self.tool_call_patterns),
            'supported_response_types': ['AIMessage', 'BaseMessage', 'Any']
        }


# Global response processor instance
_response_processor = None

def get_response_processor() -> ResponseProcessor:
    """Get the global response processor instance"""
    global _response_processor
    if _response_processor is None:
        _response_processor = ResponseProcessor()
    return _response_processor


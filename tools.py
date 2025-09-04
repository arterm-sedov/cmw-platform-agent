# tools.py - Consolidated tools
# Dependencies are included

import os
import io
import json
import uuid
import base64
import shutil
import requests
import tempfile
import urllib.parse
import numpy as np
import pandas as pd
import subprocess
import sys
import sqlite3
import cmath
import time
import re
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from typing import Any, Dict, List, Optional, Union
import chess

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Try to import pytesseract for OCR
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

# Try to import chess for chess analysis
try:
    import chess
    import chess.engine
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False
    chess = None

# Always import the tool decorator - it's essential
from langchain_core.tools import tool
# Expose Comindware Platform tool(s)
from tool_edit_or_create_text_attribute import edit_or_create_text_attribute  # noqa: F401

# Global configuration for search tools
SEARCH_LIMIT = 5  # Maximum number of results for all search tools (Tavily, Wikipedia, Arxiv)

# LangChain imports for search tools
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Warning: TavilySearch not available. Install with: pip install langchain-tavily")

# Try to import wikipedia-api as it's a common dependency
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError as e:
    WIKIPEDIA_AVAILABLE = False
    print(f"Wikipedia search requires additional dependencies. Install with: pip install wikipedia-api. Error: {str(e)}")

try:
    from langchain_community.document_loaders import WikipediaLoader
    WIKILOADER_AVAILABLE = True
except ImportError:
    WIKILOADER_AVAILABLE = False
    print("Warning: WikipediaLoader not available. Install with: pip install langchain-community")

# Try to import arxiv as it's a common dependency
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError as e:
    ARXIV_AVAILABLE = False
    print(f"Arxiv search requires additional dependencies. Install with: pip install arxiv. Error: {str(e)}")

try:
    from langchain_community.document_loaders import ArxivLoader
    ARXIVLOADER_AVAILABLE = True
except ImportError:
    ARXIVLOADER_AVAILABLE = False
    print("Warning: ArxivLoader not available. Install with: pip install langchain-community")

# Try to import Exa for AI-powered answers
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    print("Warning: Exa not available. Install with: pip install exa-py")

# Google Gemini imports for video/audio/chess understanding
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gemini not available. Install with: pip install google-genai")


# ========== GEMINI HELPER FUNCTIONS ==========
def _get_gemini_client():
    """
    Initialize and return a Gemini client with proper error handling.
    Args:
        model_name (str, optional): The Gemini model to use. If None, defaults to gemini-2.5-flash.
    Returns:
        client or None: The Gemini client if initialization succeeds, None otherwise.
    """
    if not GEMINI_AVAILABLE:
        print("Warning: Google Gemini not available. Install with: pip install google-genai")
        return None
    try:
        gemini_key = os.environ.get("GEMINI_KEY")
        if not gemini_key:
            print("Warning: GEMINI_KEY not found in environment variables.")
            return None
        client = genai.Client(api_key=gemini_key)
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {str(e)}")
        return None

def _get_gemini_response(prompt, error_prefix="Gemini", model_name="gemini-2.5-flash"):
    """
    Get a response from Gemini with proper error handling.
    Args:
        prompt: The prompt to send to Gemini
        error_prefix (str): Prefix for error messages to identify the calling context
        model_name (str, optional): The Gemini model to use.
    Returns:
        str: The Gemini response text, or an error message if the request fails.
    """
    client = _get_gemini_client()
    if not client:
        return f"{error_prefix} client not available. Check installation and API key configuration."
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error in {error_prefix.lower()} request: {str(e)}"

# ========== IMAGE PROCESSING HELPERS ==========
def encode_image(image_path: str) -> str:
    """
    Convert an image file to a base64-encoded string.

    Args:
        image_path (str): The path to the image file to encode.

    Returns:
        str: The base64-encoded string representation of the image file.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def decode_image(base64_string: str) -> Any:
    """
    Convert a base64-encoded string to a PIL Image object.

    Args:
        base64_string (str): The base64-encoded string representing the image.

    Returns:
        Any: The decoded PIL Image object.
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def save_image(image: Any, directory: str = "image_outputs") -> str:
    """
    Save a PIL Image object to disk in the specified directory and return the file path.

    Args:
        image (Any): The PIL Image object to save.
        directory (str, optional): The directory to save the image in. Defaults to "image_outputs".

    Returns:
        str: The file path where the image was saved.
    """
    os.makedirs(directory, exist_ok=True)
    image_id = str(uuid.uuid4())
    image_path = os.path.join(directory, f"{image_id}.png")
    image.save(image_path)
    return image_path

# ========== CODE INTERPRETER ==========
class CodeInterpreter:
    """
    A code interpreter for executing code in various languages (Python, Bash, SQL, C, Java) with safety and resource controls.

    Args:
        allowed_modules (list, optional): List of allowed module names for Python execution.
        max_execution_time (int, optional): Maximum execution time in seconds for code blocks.
        working_directory (str, optional): Directory for temporary files and execution context.

    Attributes:
        globals (dict): Global variables for code execution.
        temp_sqlite_db (str): Path to a temporary SQLite database for SQL code.
    """
    def __init__(self, allowed_modules=None, max_execution_time=30, working_directory=None):
        self.allowed_modules = allowed_modules or [
            "numpy", "pandas", "matplotlib", "scipy", "sklearn", 
            "math", "random", "statistics", "datetime", "collections",
            "itertools", "functools", "operator", "re", "json",
            "sympy", "networkx", "nltk", "PIL", "pytesseract", 
            "cmath", "uuid", "tempfile", "requests", "urllib"
        ]
        self.max_execution_time = max_execution_time
        self.working_directory = working_directory or os.path.join(os.getcwd()) 
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        
        # Use global imports that are already available
        self.globals = {
            "__builtins__": __builtins__,
            "np": np,
            "pd": pd,
            "Image": Image,
        }
        
        # Only add plt to globals if it's available
        if MATPLOTLIB_AVAILABLE:
            self.globals["plt"] = plt
        
        self.temp_sqlite_db = os.path.join(tempfile.gettempdir(), "code_exec.db")
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code in the specified language with safety controls.
        
        Args:
            code (str): The source code to execute
            language (str): The programming language
            
        Returns:
            Dict containing execution results, status, and outputs
        """
        try:
            if language.lower() == "python":
                return self._execute_python(code)
            elif language.lower() == "bash":
                return self._execute_bash(code)
            elif language.lower() == "sql":
                return self._execute_sql(code)
            elif language.lower() == "c":
                return self._execute_c(code)
            elif language.lower() == "java":
                return self._execute_java(code)
            else:
                return {"status": "error", "stderr": f"Unsupported language: {language}"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code with safety controls."""
        try:
            # Capture stdout and stderr
            # Create string buffers to capture output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            # Store original stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Redirect stdout/stderr to our buffers
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            
            try:
                # Create a copy of globals for this execution
                local_globals = self.globals.copy()
                local_globals['__name__'] = '__main__'
                
                # Execute the code
                exec(code, local_globals)
                
                # Get captured output
                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()
                
                # Capture any variables that might be dataframes or plots
                result = {"status": "success", "stdout": stdout_content, "stderr": stderr_content, "result": None}
                
                # Check for dataframes
                dataframes = []
                for name, value in local_globals.items():
                    if isinstance(value, pd.DataFrame):
                        dataframes.append({
                            "name": name,
                            "shape": value.shape,
                            "head": value.head().to_dict('records')
                        })
                if dataframes:
                    result["dataframes"] = dataframes
                
                # Check for plots (only if matplotlib is available)
                plots = []
                if MATPLOTLIB_AVAILABLE and plt is not None:
                    try:
                        # Save any current plots
                        if plt.get_fignums():
                            for fig_num in plt.get_fignums():
                                fig = plt.figure(fig_num)
                                plot_path = os.path.join(self.working_directory, f"plot_{fig_num}.png")
                                fig.savefig(plot_path)
                                plots.append(plot_path)
                                plt.close(fig)
                    except Exception as plot_error:
                        # If plot handling fails, just continue without plots
                        print(f"Warning: Plot handling failed: {plot_error}")
                if plots:
                    result["plots"] = plots
                
                return result
                
            finally:
                # Restore original stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                stdout_buffer.close()
                stderr_buffer.close()
            
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_bash(self, code: str) -> Dict[str, Any]:
        """Execute Bash code."""
        try:
            result = subprocess.run(
                code, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=self.max_execution_time
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_sql(self, code: str) -> Dict[str, Any]:
        """Execute SQL code using SQLite."""
        try:
            conn = sqlite3.connect(self.temp_sqlite_db)
            cursor = conn.cursor()
            
            # Execute SQL
            cursor.execute(code)
            
            # Fetch results if it's a SELECT
            if code.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                result = {"status": "success", "results": results, "columns": columns}
            else:
                conn.commit()
                result = {"status": "success", "message": f"Executed: {code}"}
            
            conn.close()
            return result
            
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_c(self, code: str) -> Dict[str, Any]:
        """Execute C code by compiling and running."""
        try:
            # Create temporary C file
            c_file = os.path.join(self.working_directory, "temp_code.c")
            with open(c_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ["gcc", "-o", os.path.join(self.working_directory, "temp_program"), c_file],
                capture_output=True,
                text=True
            )
            
            if compile_result.returncode != 0:
                return {"status": "error", "stderr": f"Compilation failed: {compile_result.stderr}"}
            
            # Run
            run_result = subprocess.run(
                [os.path.join(self.working_directory, "temp_program")],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time
            )
            
            return {
                "status": "success",
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "returncode": run_result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_java(self, code: str) -> Dict[str, Any]:
        """Execute Java code by compiling and running."""
        try:
            # Create temporary Java file
            java_file = os.path.join(self.working_directory, "TempCode.java")
            with open(java_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ["javac", java_file],
                capture_output=True,
                text=True
            )
            
            if compile_result.returncode != 0:
                return {"status": "error", "stderr": f"Compilation failed: {compile_result.stderr}"}
            
            # Run
            run_result = subprocess.run(
                ["java", "-cp", self.working_directory, "TempCode"],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time
            )
            
            return {
                "status": "success",
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "returncode": run_result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}

# Create a global instance for use by tools
interpreter_instance = CodeInterpreter()

@tool
def execute_code_multilang(code: str, language: str = "python") -> str:
    """Execute code in multiple languages (Python, Bash, SQL, C, Java) and return results.

    Args:
        code (str): The source code to execute.
        language (str): The language of the code. Supported: "python", "bash", "sql", "c", "java".

    Returns:
        A string summarizing the execution results (stdout, stderr, errors, plots, dataframes if any).
    """
    supported_languages = ["python", "bash", "sql", "c", "java"]
    language = language.lower()

    if language not in supported_languages:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "execute_code_multilang",
            "error": f"❌ Unsupported language: {language}. Supported languages are: {', '.join(supported_languages)}"
        })

    result = interpreter_instance.execute_code(code, language=language)

    response = []

    if result["status"] == "success":
        response.append(f"✅ Code executed successfully in **{language.upper()}**")

        if result.get("stdout"):
            response.append(
                "\n**Standard Output:**\n```\n" + result["stdout"].strip() + "\n```"
            )

        if result.get("stderr"):
            response.append(
                "\n**Standard Error (if any):**\n```\n"
                + result["stderr"].strip()
                + "\n```"
            )

        if result.get("result") is not None:
            response.append(
                "\n**Execution Result:**\n```\n"
                + str(result["result"]).strip()
                + "\n```"
            )

        if result.get("dataframes"):
            for df_info in result["dataframes"]:
                response.append(
                    f"\n**DataFrame `{df_info['name']}` (Shape: {df_info['shape']})**"
                )
                df_preview = pd.DataFrame(df_info["head"])
                response.append("First 5 rows:\n```\n" + str(df_preview) + "\n```")

        if result.get("plots"):
            response.append(
                f"\n**Generated {len(result['plots'])} plot(s)** (Image data returned separately)"
            )

    else:
        response.append(f"❌ Code execution failed in **{language.upper()}**")
        if result.get("stderr"):
            response.append(
                "\n**Error Log:**\n```\n" + result["stderr"].strip() + "\n```"
            )

    return json.dumps({
        "type": "tool_response",
        "tool_name": "execute_code_multilang",
        "result": "\n".join(response)
    })

# ========== MATH TOOLS ==========
@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers and return the result.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The product of a and b.
    """
    return a * b

@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers and return the result.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of a and b.
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """
    Subtract the second number from the first and return the result.

    Args:
        a (float): The number to subtract from.
        b (float): The number to subtract.

    Returns:
        float: The result of a - b.
    """
    return a - b

@tool
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second and return the result.

    Args:
        a (float): The numerator.
        b (float): The denominator. Must not be zero.

    Returns:
        float: The quotient of a and b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """
    Compute the modulus (remainder) of two integers.

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        int: The remainder when a is divided by b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a % b

@tool
def power(a: float, b: float) -> float:
    """
    Raise the first number to the power of the second and return the result.

    Args:
        a (float): The base number.
        b (float): The exponent.

    Returns:
        float: a raised to the power of b.
    """
    return a ** b

@tool
def square_root(a: float) -> float:
    """
    Compute the square root of a number. Returns a complex number if input is negative.

    Args:
        a (float): The number to compute the square root of.

    Returns:
        float or complex: The square root of a. If a < 0, returns a complex number.
    """
    if a >= 0:
        return a ** 0.5
    return cmath.sqrt(a)

# ========== WEB/SEARCH TOOLS ==========
@tool
def wiki_search(input: str) -> str:
    """
    Search Wikipedia for a query and return up to 3 results as formatted text.

    Args:
        input (str): The search query string for Wikipedia.

    Returns:
        str: Formatted search results from Wikipedia with source information and content.
    """
    try:
        if not WIKILOADER_AVAILABLE:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "wiki_search",
                "error": "Wikipedia search not available. Install with: pip install langchain-community"
            })
        search_docs = WikipediaLoader(query=input, load_max_docs=SEARCH_LIMIT).load()
        formatted_results = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
                for doc in search_docs
            ]
        )
        return json.dumps({
            "type": "tool_response",
            "tool_name": "wiki_search",
            "wiki_results": formatted_results
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "wiki_search",
            "error": f"Error in Wikipedia search: {str(e)}"
        })

@tool
def web_search(input: str) -> str:
    """
    Search the web using Tavily for a query and return up to 3 results as formatted text.
    
    Tavily is a search API that provides real-time web search results. This tool is useful for:
    - Finding current information about recent events
    - Searching for specific facts, statistics, or data
    - Getting up-to-date information from various websites
    - Researching topics that may not be covered in Wikipedia or academic papers

    Args:
        input (str): The search query string to search for on the web.

    Returns:
        str: Formatted search results from Tavily with source URLs and content snippets.
             Returns an error message if Tavily is not available or if the search fails.

    """
    if not TAVILY_AVAILABLE:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "web_search",
            "error": "Tavily search not available. Install with: pip install langchain-tavily"
        })
    try:
        if not os.environ.get("TAVILY_API_KEY"):
            return json.dumps({
                "type": "tool_response",
                "tool_name": "web_search",
                "error": "TAVILY_API_KEY not found in environment variables. Please set it in your .env file."
            })
        search_result = TavilySearch(max_results=SEARCH_LIMIT).invoke(input)
        
        # Handle different response types
        if isinstance(search_result, str):
            # If Tavily returned a string (error message or direct answer)
            return json.dumps({
                "type": "tool_response",
                "tool_name": "web_search",
                "web_results": search_result
            })
        elif isinstance(search_result, list):
            # If Tavily returned a list of Document objects
            formatted_results = "\n\n---\n\n".join(
                [
                    f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
                    for doc in search_result
                ]
            )
            return json.dumps({
                "type": "tool_response",
                "tool_name": "web_search",
                "web_results": formatted_results
            })
        else:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "web_search",
                    "web_results": str(search_result)
            })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "web_search",
            "error": f"Error in web search: {str(e)}"
        })

@tool
def arxiv_search(input: str) -> str:
    """
    Search Arxiv for academic papers and return up to 3 results as formatted text.

    Args:
        input (str): The search query string for academic papers.

    Returns:
        str: Formatted search results from Arxiv with paper metadata and abstracts.
    """
    try:
        if not ARXIVLOADER_AVAILABLE:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "arxiv_search",
                "error": "Arxiv search not available. Install with: pip install langchain-community"
            })
        search_docs = ArxivLoader(query=input, load_max_docs=SEARCH_LIMIT).load()
        formatted_results = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
                for doc in search_docs
            ]
        )
        return json.dumps({
            "type": "tool_response",
            "tool_name": "arxiv_search",
            "arxiv_results": formatted_results
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "arxiv_search",
            "error": f"Error in Arxiv search: {str(e)}"
        })

# @tool
# def exa_ai_helper(question: str) -> str:
#     """
#     Prefer web_search_deep_research_exa_ai. It is smarter, and gives more researched results.
#     Smart AI web-search engine. Gives web references.
#     Get direct answers + web references.
#     Do not ask me about attached files or video/audio analysis.
        
#     This tool is particularly useful when:
#     - You need authoritative, up-to-date information on a topic
#     - You want to double-check your own knowledge or reasoning
#     - You're dealing with complex questions that require multiple sources
#     - You need citations and sources to back up your answer
#     - You're unsure about the accuracy of your response
    
#     The tool performs an Exa search and uses an LLM to generate either:
#     - A direct answer for specific queries (e.g., "What is the capital of France?" returns "Paris")
#     - A detailed summary with citations for open-ended queries (e.g., "What is the state of AI in healthcare?")
    
#     WARNING: Always judge yourself and use additional tools for research.
    
#     Args:
#         question (str): The question to get an answer for and search results. Can be specific or open-ended.
    
#     Returns:
#         str: A well-researched answer with citations and sources, or an error message.
    
#     """
#     if not EXA_AVAILABLE:
#         return json.dumps({
#             "type": "tool_response",
#             "tool_name": "exa_ai_helper",
#             "error": "Exa AI Helper not available. Install with: pip install exa-py"
#         })
#     try:
#         exa_api_key = os.environ.get("EXA_API_KEY")
#         if not exa_api_key:
#             return json.dumps({
#                 "type": "tool_response",
#                 "tool_name": "exa_ai_helper",
#                 "error": "EXA_API_KEY not found in environment variables. Please set it in your .env file."
#             })
#         exa = Exa(exa_api_key)
#         result = exa.stream_answer(
#             question,
#             text=True,
#         )
#         answer_parts = []
#         for chunk in result:
#             # If chunk is a StreamChunk, extract its text/content
#             if hasattr(chunk, 'text'):
#                 answer_parts.append(chunk.text)
#             elif isinstance(chunk, str):
#                 answer_parts.append(chunk)
#             else:
#                 answer_parts.append(str(chunk))
#         full_answer = ''.join(answer_parts)
#         return json.dumps({
#             "type": "tool_response",
#             "tool_name": "exa_ai_helper",
#             "answer": full_answer
#         })
#     except Exception as e:
#         return json.dumps({
#             "type": "tool_response",
#             "tool_name": "exa_ai_helper",
#             "error": f"Error getting AI Helper answer: {str(e)}"
#         })

# ========== FILE/DATA TOOLS ==========
@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save the provided content to a file and return the file path.

    Args:
        content (str): The content to write to the file.
        filename (str, optional): The name of the file. If not provided, a random file name is generated.

    Returns:
        str: The file path where the content was saved.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return json.dumps({
        "type": "tool_response",
        "tool_name": "save_and_read_file",
        "result": f"File saved to {filepath}. You can read this file to process its contents."
    })

@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location. Returns the file path.

    Args:
        url (str): The URL of the file to download.
        filename (str, optional): The name of the file. If not provided, a name is inferred or generated.

    Returns:
        str: The file path where the file was downloaded.
    """
    try:
        if not filename:
            from urllib.parse import urlparse
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return json.dumps({
            "type": "tool_response",
            "tool_name": "download_file_from_url",
            "result": f"File downloaded to {filepath}. You can read this file to process its contents."
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "download_file_from_url",
            "error": f"Error downloading file: {str(e)}"
        })

@tool
def get_task_file(task_id: str, file_name: str) -> str:
    """
    Download a file associated with a given task_id from the evaluation API, with a local fallback.
    
    This tool is used to download files that are part of CMW Platform Agent benchmark tasks.
    It first tries to download from the evaluation API, and if that fails
    (e.g., due to network issues or rate limits),
    it falls back to local files in the 'files' directory.
    The file is always saved to a 'downloads' directory.

    Args:
        task_id (str): The task ID for the file to download.
        file_name (str): The name of the file to download.

    Returns:
        str: The absolute file path where the file was downloaded, or an error message if not found.
    """
    directory_name = "downloads"
    os.makedirs(directory_name, exist_ok=True)
    try:
        # Try to download from evaluation API
        evaluation_api_base_url = os.environ.get("EVALUATION_API_BASE_URL", "https://api.gaia-benchmark.com")
        response = requests.get(f"{evaluation_api_base_url}/files/{task_id}", timeout=15)
        response.raise_for_status()
        filepath = os.path.join(directory_name, file_name)
        with open(filepath, 'wb') as file:
            file.write(response.content)
        return json.dumps({
            "type": "tool_response",
            "tool_name": "get_task_file",
            "result": os.path.abspath(filepath)
        })
    except Exception as e:
        # Fallback to local files
        try:
            local_filepath = os.path.join("files", file_name)
            if os.path.exists(local_filepath):
                filepath = os.path.join(directory_name, file_name)
                shutil.copy2(local_filepath, filepath)
                return json.dumps({
                    "type": "tool_response",
                    "tool_name": "get_task_file",
                    "result": os.path.abspath(filepath)
                })
            else:
                return json.dumps({
                    "type": "tool_response",
                    "tool_name": "get_task_file",
                    "error": f"Error: File {file_name} not found locally or via API"
                })
        except Exception as local_error:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "get_task_file",
                "error": f"Error downloading file: {str(e)}. Local fallback also failed: {str(local_error)}"
            })

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using OCR (pytesseract) and return the extracted text.

    Args:
        image_path (str): The path to the image file to process.

    Returns:
        str: The extracted text, or an error message if extraction fails.
    """
    try:
        image = Image.open(image_path)
        if PYTESSERACT_AVAILABLE:
            text = pytesseract.image_to_string(image)
        else:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "extract_text_from_image",
                "error": "OCR not available. Install with: pip install pytesseract"
            })
        return json.dumps({
            "type": "tool_response",
            "tool_name": "extract_text_from_image",
            "result": f"Extracted text from image:\n\n{text}"
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "extract_text_from_image",
            "error": f"Error extracting text from image: {str(e)}"
        })

@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and return summary statistics and column info.

    Args:
        file_path (str): The path to the CSV file.
        query (str): A question or description of the analysis to perform (currently unused).

    Returns:
        str: Summary statistics and column information, or an error message if analysis fails.
    """
    try:
        df = pd.read_csv(file_path)
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        result += "Summary statistics:\n"
        result += str(df.describe())
        return json.dumps({
            "type": "tool_response",
            "tool_name": "analyze_csv_file",
            "result": result
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "analyze_csv_file",
            "error": f"Error analyzing CSV file: {str(e)}"
        })

@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and return summary statistics and column info.

    Args:
        file_path (str): The path to the Excel file.
        query (str): A question or description of the analysis to perform (currently unused).

    Returns:
        str: Summary statistics and column information, or an error message if analysis fails.
    """
    try:
        df = pd.read_excel(file_path)
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        result += "Summary statistics:\n"
        result += str(df.describe())
        return json.dumps({
            "type": "tool_response",
            "tool_name": "analyze_excel_file",
            "result": result
        })
    except Exception as e:
        # Enhanced error reporting: print columns and head if possible
        try:
            df = pd.read_excel(file_path)
            columns = list(df.columns)
            head = df.head().to_dict('records')
            error_details = f"Error analyzing Excel file: {str(e)}\nColumns: {columns}\nHead: {head}"
        except Exception as inner_e:
            error_details = f"Error analyzing Excel file: {str(e)}\nAdditionally, failed to read columns/head: {str(inner_e)}"
        return json.dumps({
            "type": "tool_response",
            "tool_name": "analyze_excel_file",
            "error": error_details
        })

# ========== IMAGE ANALYSIS/GENERATION TOOLS ==========
@tool
def analyze_image(image_base64: str) -> str:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview) from a base64-encoded image string.

    Args:
        image_base64 (str): The base64-encoded string of the image to analyze.

    Returns:
        str: JSON string with analysis results including dimensions, mode, color_analysis, and thumbnail.
    """
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode
        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            avg_colors = arr.mean(axis=(0, 1))
            dominant = ["Red", "Green", "Blue"][np.argmax(avg_colors[:3])]
            brightness = avg_colors.mean()
            color_analysis = {
                "average_rgb": avg_colors.tolist(),
                "brightness": brightness,
                "dominant_color": dominant,
            }
        else:
            color_analysis = {"note": f"No color analysis for mode {mode}"}
        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)
        result = {
            "dimensions": (width, height),
            "mode": mode,
            "color_analysis": color_analysis,
            "thumbnail": thumbnail_base64,
        }
        return json.dumps({
            "type": "tool_response",
            "tool_name": "analyze_image",
            "result": result
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "analyze_image",
            "error": str(e)
        }, indent=2)

@tool
def transform_image(image_base64: str, operation: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Transform an image using various operations like resize, rotate, filter, etc.

    Args:
        image_base64 (str): The base64-encoded string of the image to transform.
        operation (str): The transformation operation to apply.
        params (Dict[str, Any], optional): Parameters for the transformation.

    Returns:
        str: JSON string with the transformed image as base64 or error message.
    """
    try:
        img = decode_image(image_base64)
        params = params or {}
        if operation == "resize":
            width = params.get("width", img.width)
            height = params.get("height", img.height)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        elif operation == "rotate":
            angle = params.get("angle", 0)
            img = img.rotate(angle, expand=True)
        elif operation == "flip":
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif operation == "blur":
            radius = params.get("radius", 2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        elif operation == "sharpen":
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        elif operation == "brightness":
            factor = params.get("factor", 1.0)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        elif operation == "contrast":
            factor = params.get("factor", 1.0)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        else:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "transform_image",
                "error": f"Unsupported operation: {operation}"
            }, indent=2)
        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return json.dumps({
            "type": "tool_response",
            "tool_name": "transform_image",
            "transformed_image": result_base64
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "transform_image",
            "error": str(e)
        }, indent=2)

@tool
def draw_on_image(image_base64: str, drawing_type: str, params: Dict[str, Any]) -> str:
    """
    Draw shapes, text, or other elements on an image.

    Args:
        image_base64 (str): The base64-encoded string of the image to draw on.
        drawing_type (str): The type of drawing to perform.
        params (Dict[str, Any]): Parameters for the drawing operation.

    Returns:
        str: JSON string with the modified image as base64 or error message.
    """
    try:
        img = decode_image(image_base64)
        draw = ImageDraw.Draw(img)
        if drawing_type == "text":
            text = params.get("text", "")
            position = params.get("position", (10, 10))
            color = params.get("color", "black")
            size = params.get("size", 20)
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except:
                font = ImageFont.load_default()
            draw.text(position, text, fill=color, font=font)
        elif drawing_type == "rectangle":
            coords = params.get("coords", [10, 10, 100, 100])
            color = params.get("color", "red")
            width = params.get("width", 2)
            draw.rectangle(coords, outline=color, width=width)
        elif drawing_type == "circle":
            center = params.get("center", (50, 50))
            radius = params.get("radius", 30)
            color = params.get("color", "blue")
            width = params.get("width", 2)
            bbox = [center[0] - radius, center[1] - radius, 
                   center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, outline=color, width=width)
        elif drawing_type == "line":
            start = params.get("start", (10, 10))
            end = params.get("end", (100, 100))
            color = params.get("color", "green")
            width = params.get("width", 2)
            draw.line([start, end], fill=color, width=width)
        else:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "draw_on_image",
                "error": f"Unsupported drawing type: {drawing_type}"
            }, indent=2)
        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return json.dumps({
            "type": "tool_response",
            "tool_name": "draw_on_image",
            "modified_image": result_base64
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "draw_on_image",
            "error": str(e)
        }, indent=2)

@tool
def generate_simple_image(image_type: str, width: int = 500, height: int = 500, 
                         params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate simple images like gradients, solid colors, checkerboard, or noise patterns.

    Args:
        image_type (str): The type of image to generate.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        params (Dict[str, Any], optional): Additional parameters for image generation.

    Returns:
        str: JSON string with the generated image as base64 or error message.
    """
    try:
        params = params or {}
        if image_type == "solid":
            color = params.get("color", (255, 255, 255))
            img = Image.new("RGB", (width, height), color)
        elif image_type == "gradient":
            start_color = params.get("start_color", (255, 0, 0))
            end_color = params.get("end_color", (0, 0, 255))
            direction = params.get("direction", "horizontal")
            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)
            if direction == "horizontal":
                for x in range(width):
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * x / width)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * x / width)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * x / width)
                    draw.line([(x, 0), (x, height)], fill=(r, g, b))
            else:
                for y in range(height):
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * y / height)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * y / height)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * y / height)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))
        elif image_type == "noise":
            noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise_array, "RGB")
        elif image_type == "checkerboard":
            square_size = params.get("square_size", 50)
            color1 = params.get("color1", "white")
            color2 = params.get("color2", "black")
            img = Image.new("RGB", (width, height))
            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    color = color1 if ((x // square_size) + (y // square_size)) % 2 == 0 else color2
                    for dy in range(square_size):
                        for dx in range(square_size):
                            if x + dx < width and y + dy < height:
                                img.putpixel((x + dx, y + dy), color)
        else:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "generate_simple_image",
                "error": f"Unsupported image_type {image_type}"
            }, indent=2)
        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return json.dumps({
            "type": "tool_response",
            "tool_name": "generate_simple_image",
            "generated_image": result_base64
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "generate_simple_image",
            "error": str(e)
        }, indent=2)

@tool
def combine_images(images_base64: List[str], operation: str, 
                  params: Optional[Dict[str, Any]] = None) -> str:
    """
    Combine multiple images using various operations (collage, stack, blend, horizontal, vertical, overlay, etc.).

    Args:
        images_base64 (List[str]): List of base64-encoded image strings.
        operation (str): The combination operation to perform.
        params (Dict[str, Any], optional): Parameters for the combination.

    Returns:
        str: JSON string with the combined image as base64 or error message.
    """
    try:
        if len(images_base64) < 2:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "combine_images",
                "error": "At least 2 images required for combination"
            }, indent=2)
        images = [decode_image(b64) for b64 in images_base64]
        params = params or {}
        if operation == "horizontal":
            # Combine images side by side
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            result = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in images:
                result.paste(img, (x_offset, 0))
                x_offset += img.width
        elif operation == "vertical":
            # Stack images vertically
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)
            result = Image.new("RGB", (max_width, total_height))
            y_offset = 0
            for img in images:
                result.paste(img, (0, y_offset))
                y_offset += img.height
        elif operation == "overlay":
            # Overlay images on top of each other
            base_img = images[0]
            for overlay_img in images[1:]:
                if overlay_img.size != base_img.size:
                    overlay_img = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)
                base_img = Image.alpha_composite(base_img.convert("RGBA"), overlay_img.convert("RGBA"))
            result = base_img.convert("RGB")
        elif operation == "stack":
            # Original stack operation with direction parameter
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                total_width = sum(img.width for img in images)
                max_height = max(img.height for img in images)
                result = Image.new("RGB", (total_width, max_height))
                x = 0
                for img in images:
                    result.paste(img, (x, 0))
                    x += img.width
            else:
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)
                result = Image.new("RGB", (max_width, total_height))
                y = 0
                for img in images:
                    result.paste(img, (0, y))
                    y += img.height
        else:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "combine_images",
                "error": f"Unsupported combination operation: {operation}"
            }, indent=2)
        result_path = save_image(result)
        result_base64 = encode_image(result_path)
        return json.dumps({
            "type": "tool_response",
            "tool_name": "combine_images",
            "combined_image": result_base64
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "combine_images",
            "error": str(e)
        }, indent=2)

# ========== VIDEO/AUDIO UNDERSTANDING TOOLS ==========
@tool
def understand_video(youtube_url: str, prompt: str, system_prompt: str = None) -> str:
    """
    Analyze a YouTube video using Google Gemini's video understanding capabilities.
    
    This tool can understand video content, extract information, and answer questions
    about what happens in the video.
    It uses the Gemini API and requires the GEMINI_KEY environment variable to be set.
    
    Args:
        youtube_url (str): The URL of the YouTube video to analyze.
        prompt (str): A question or request regarding the video content.
        system_prompt (str, optional): System prompt for formatting guidance.
    
    Returns:
        str: Analysis of the video content based on the prompt, or error message.
    """
    try:
        client = _get_gemini_client()
        
        # Create enhanced prompt with system prompt if provided
        if system_prompt:
            enhanced_prompt = f"{system_prompt}\n\nAnalyze the video at {youtube_url} and answer the following question:\n{prompt}\n\nProvide your answer in the required FINAL ANSWER format."
        else:
            enhanced_prompt = prompt
        
        video_description = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=youtube_url)),
                    types.Part(text=enhanced_prompt)
                ]
            )
        )
        return json.dumps({
            "type": "tool_response",
            "tool_name": "understand_video",
            "result": video_description.text
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "understand_video",
            "error": f"Error understanding video: {str(e)}"
        })

@tool
def understand_audio(file_path: str, prompt: str, system_prompt: str = None) -> str:
    """
    Analyze an audio file using Google Gemini's audio understanding capabilities.
    
    This tool can transcribe audio, understand spoken content, and answer questions
    about the audio content.
    It uses the Gemini API and requires the GEMINI_KEY environment variable to be set.
    The audio file is uploaded to Gemini and then analyzed with the provided prompt.
    
    Args:
        file_path (str): The path to the local audio file to analyze, or base64 encoded audio data.
        prompt (str): A question or request regarding the audio content.
        system_prompt (str, optional): System prompt for formatting guidance.
    
    Returns:
        str: Analysis of the audio content based on the prompt, or error message.
    """
    try:
        client = _get_gemini_client()
        
        # Check if file_path is base64 data or actual file path
        if file_path.startswith('/') or os.path.exists(file_path):
            # It's a file path
            mp3_file = client.files.upload(file=file_path)
        else:
            # Assume it's base64 data
            try:
                # Decode base64 and create temporary file
                audio_data = base64.b64decode(file_path)
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                try:
                    mp3_file = client.files.upload(file=temp_file_path)
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
            except Exception as decode_error:
                return json.dumps({
                    "type": "tool_response",
                    "tool_name": "understand_audio",
                    "error": f"Error processing audio data: {str(decode_error)}. Expected base64 encoded audio data or valid file path."
                })
        
        # Create enhanced prompt with system prompt if provided
        if system_prompt:
            enhanced_prompt = f"{system_prompt}\n\nAnalyze the audio file and answer the following question:\n{prompt}\n\nProvide your answer in the required FINAL ANSWER format."
        else:
            enhanced_prompt = prompt
        
        contents = [enhanced_prompt, mp3_file]
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents
            )
            return json.dumps({
                "type": "tool_response",
                "tool_name": "understand_audio",
                "result": response.text
            })
        except Exception as e:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "understand_audio",
                "error": f"Error in audio understanding request: {str(e)}"
            })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "understand_audio",
            "error": f"Error understanding audio: {str(e)}"
        })

# ========== CHESS TOOLS ==========
def _convert_chess_move_internal(piece_placement: str, move: str) -> str:
    """
    Internal function to convert chess moves from coordinate notation to algebraic notation.
    Uses Google Gemini to convert chess moves between different notations.
    Coordinate notation uses square names (e.g., "e2e4"), while algebraic notation
    uses piece symbols and square names (e.g., "e4", "Nf3", "O-O").
    The function constructs a prompt for Gemini and expects 
    only the algebraic notation as output, with no extra commentary.
    """
    prompt = f"""
    Convert this chess move from coordinate notation to algebraic notation.
    
    Piece placement: {piece_placement}
    Move in coordinate notation: {move}
    
    Return only the algebraic notation (e.g., "e4", "Nf3", "O-O", "Qxd5", etc.)
    """
    return json.dumps({
        "type": "tool_response",
        "tool_name": "convert_chess_move",
        "result": _get_gemini_response(prompt, "Chess move conversion", "gemini-2.5-pro")
    })

@tool
def convert_chess_move(piece_placement: str, move: str) -> str:
    """
    Convert a chess move from coordinate notation to algebraic notation using Google Gemini.
    
    This tool uses Google Gemini to convert chess moves between different notations.
    Coordinate notation uses square names (e.g., "e2e4"), while algebraic notation
    uses piece symbols and square names (e.g., "e4", "Nf3", "O-O").
    The function constructs a prompt for Gemini and expects 
    only the algebraic notation as output, with no extra commentary.
    
    Args:
        piece_placement (str): The chess piece placement in plain text or FEN format.
        move (str): The move in coordinate notation (e.g., "e2e4").

    Returns:
        str: The move in algebraic notation, or error message.
    """
    move_message = (
        f"Convert this chess move from coordinate notation to algebraic "
        f"notation: {move}. Use the following piece placement: {piece_placement}. "
        f"Do not provide any additional thinking or commentary in the response, "
        f"just the algebraic notation only."
    )
    return json.dumps({
        "type": "tool_response",
        "tool_name": "convert_chess_move",
        "result": _get_gemini_response(move_message, "Chess move conversion", "gemini-2.5-pro")
    })

# --- Lichess Cloud Evaluation API Helper ---
def _get_lichess_cloud_eval_candidates(fen: str, depth: int = 15) -> list:
    """
    Query the Lichess Cloud Evaluation API for candidate moves.
    Returns a list of dicts, each with move, full_line, cp, mate, depth, multipv, and explanation.
    """
    candidates = []
    chess_eval_url = os.environ.get("CHESS_EVAL_URL", "https://lichess.org/api/cloud-eval")
    url = f"{chess_eval_url}?fen={urllib.parse.quote(fen)}&depth={depth}"
    headers = {}
    lichess_key = os.environ.get("LICHESS_KEY")
    if lichess_key:
        headers["Authorization"] = f"Bearer {lichess_key}"
    try:
        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'pvs' in data and len(data['pvs']) > 0:
                for pv in data['pvs']:
                    moves_string = pv.get('moves', '')
                    if moves_string:
                        first_move = moves_string.split()[0]
                        candidates.append({
                            "source": "lichess_api",
                            "move": first_move,
                            "full_line": moves_string,
                            "cp": pv.get("cp"),
                            "mate": pv.get("mate"),
                            "depth": pv.get("depth"),
                            "multipv": pv.get("multipv"),
                            "explanation": "Move suggested by Lichess Cloud Evaluation API (principal variation)."
                        })
                    else:
                        candidates.append({
                            "source": "lichess_api",
                            "move": None,
                            "explanation": "Lichess API returned a PV with no moves."
                        })
            else:
                candidates.append({
                    "source": "lichess_api",
                    "move": None,
                    "explanation": "Lichess API returned no pvs data in response."
                })
        else:
            candidates.append({
                "source": "lichess_api",
                "move": None,
                "explanation": f"Lichess API error: HTTP {response.status_code}"
            })
    except Exception as e:
        candidates.append({
            "source": "lichess_api",
            "move": None,
            "explanation": f"Lichess API exception: {str(e)}"
        })
    return candidates

# --- Stockfish Online API Helper ---
def _get_stockfish_online_candidate(fen: str, depth: int = 15, _retry: int = 0) -> dict:
    """
    Query the Stockfish Online API for the best move for a given FEN.
    Returns a dict with move, full_line, evaluation (cp), mate, and explanation.
    Retries once on timeout (443) errors, waits 30 seconds before retrying, then fails gracefully.
    """
    api_url = "https://stockfish.online/api/s/v2.php"
    params = {'fen': fen, 'depth': depth}
    try:
        response = requests.get(api_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                bestmove = data.get('bestmove', '')
                move = None
                if bestmove:
                    move_parts = bestmove.split()
                    if len(move_parts) >= 2 and move_parts[0] == 'bestmove':
                        move = move_parts[1]
                # Extract useful fields
                return {
                    "source": "stockfish_online_api",
                    "move": move,
                    "full_line": data.get("continuation"),
                    "cp": data.get("evaluation"),
                    "mate": data.get("mate"),
                    "explanation": "Move suggested by Stockfish Online API v2." if move else f"Stockfish Online API error: {data}"
                }
            else:
                return {
                    "source": "stockfish_online_api",
                    "move": None,
                    "explanation": f"Stockfish API failed: {data.get('data', 'Unknown error')}"
                }
        else:
            return {
                "source": "stockfish_online_api",
                "move": None,
                "explanation": f"Stockfish API HTTP error: {response.status_code}"
            }
    except Exception as e:
        # Simple retry on timeout/443 error, then fail gracefully
        if _retry < 1 and ("443" in str(e) or "timed out" in str(e).lower() or "timeout" in str(e).lower()):
            time.sleep(30)
            return _get_stockfish_online_candidate(fen, depth, _retry=_retry+1)
        return {
            "source": "stockfish_online_api",
            "move": None,
            "explanation": f"Stockfish API exception: {str(e)}"
        }

def _get_python_chess_stockfish_candidate(fen: str, depth: int = 15) -> dict:
    """
    Try to get a move using local python-chess Stockfish engine. If not available, fallback to Stockfish Online API.
    Returns a dict with move and explanation.
    """
    try:
        if 'CHESS_AVAILABLE' in globals() and CHESS_AVAILABLE:
            import chess
            import chess.engine
            board = chess.Board(fen)
            try:
                engine = chess.engine.SimpleEngine.popen_uci("stockfish")
                result = engine.play(board, chess.engine.Limit(time=2.0))
                engine.quit()
                if result.move:
                    move = chess.square_name(result.move.from_square) + chess.square_name(result.move.to_square)
                    return {
                        "source": "python_chess_stockfish",
                        "move": move,
                        "explanation": "Move suggested by local Stockfish engine via python-chess."
                    }
                else:
                    return {
                        "source": "python_chess_stockfish",
                        "move": None,
                        "explanation": "python-chess Stockfish engine returned no move."
                    }
            except FileNotFoundError as e:
                # Fallback to Stockfish Online API if local binary is missing
                online = _get_stockfish_online_candidate(fen, depth)
                online["source"] = "python_chess_stockfish (online fallback)"
                online["explanation"] = "Local Stockfish not found, used Stockfish Online API as fallback. " + online.get("explanation", "")
                return online
            except Exception as e:
                return {
                    "source": "python_chess_stockfish",
                    "move": None,
                    "explanation": f"python-chess Stockfish engine exception: {str(e)}"
                }
        else:
            return {
                "source": "python_chess_stockfish",
                "move": None,
                "explanation": "python-chess or Stockfish engine not available."
            }
    except Exception as e:
        return {
            "source": "python_chess_stockfish",
            "move": None,
            "explanation": f"python-chess Stockfish engine import/availability exception: {str(e)}"
        }

# --- Main Internal Move Candidate Function ---
def _get_best_chess_move_internal(fen: str) -> dict:
    """
    Internal function to get the best chess move for a given FEN position.
    Tries multiple sources (Lichess, Stockfish Online, python-chess, heuristics) and returns all candidates with explanations for LLM selection.
    Returns a Python dict, not a JSON string.
    """
    move_candidates = []
    # 1. Lichess API (all PVs)
    move_candidates.extend(_get_lichess_cloud_eval_candidates(fen))
    # 2. Stockfish Online API (single best move)
    move_candidates.append(_get_stockfish_online_candidate(fen))
    # 3. python-chess local engine, with online fallback
    move_candidates.append(_get_python_chess_stockfish_candidate(fen))
    # 4. _get_best_move_simple_heuristic
    try:
        heuristic_move = _get_best_move_simple_heuristic(fen)
        move = None
        if isinstance(heuristic_move, str) and len(heuristic_move) in [4, 5]:
            move = heuristic_move
        move_candidates.append({
            "source": "simple_heuristic",
            "move": move,
            "explanation": "Move suggested by simple FEN-based heuristic." if move else f"Heuristic error: {heuristic_move}"
        })
    except Exception as e:
        move_candidates.append({
            "source": "simple_heuristic",
            "move": None,
            "explanation": f"Simple heuristic exception: {str(e)}"
        })
    # 5. _evaluate_moves_simple
    try:
        if 'CHESS_AVAILABLE' in globals() and CHESS_AVAILABLE:
            import chess
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            best_move = _evaluate_moves_simple(board, legal_moves)
            move = None
            if best_move:
                move = chess.square_name(best_move.from_square) + chess.square_name(best_move.to_square)
            move_candidates.append({
                "source": "evaluate_moves_simple",
                "move": move,
                "explanation": "Move suggested by simple move evaluation (captures, checks, center, development)." if move else "No move found by simple evaluation."
            })
    except Exception as e:
        move_candidates.append({
            "source": "evaluate_moves_simple",
            "move": None,
            "explanation": f"Simple evaluation exception: {str(e)}"
        })
    return {
        "fen": fen,
        "candidates": move_candidates
    }

def _get_best_move_fallback(fen: str) -> str:
    """
    Fallback function to get best move when Lichess API returns 404.
    Uses alternative APIs, local chess engine, and intelligent heuristics.
    """
    try:
        # Try alternative chess API (Stockfish Online API v2)
        try:
            stockfish_result = _try_stockfish_online_api_v2(fen)
            if not stockfish_result.startswith("Error"):
                return stockfish_result
        except:
            pass
        
        # Try using Stockfish via python-chess if available
        try:
            if CHESS_AVAILABLE:
                board = chess.Board(fen)
                
                # Use Stockfish if available
                try:
                    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
                    result = engine.play(board, chess.engine.Limit(time=2.0))
                    engine.quit()
                    if result.move:
                        return chess.square_name(result.move.from_square) + chess.square_name(result.move.to_square)
                except:
                    pass
                
                # Fallback: use legal moves and simple evaluation
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    # Try to find a good move using simple evaluation
                    best_move = _evaluate_moves_simple(board, legal_moves)
                    if best_move:
                        return chess.square_name(best_move.from_square) + chess.square_name(best_move.to_square)
                    else:
                        # Return first legal move as fallback
                        move = legal_moves[0]
                        return chess.square_name(move.from_square) + chess.square_name(move.to_square)
                else:
                    return json.dumps({
                        "type": "tool_response",
                        "tool_name": "get_best_chess_move",
                        "error": "Error: No legal moves available"
                    })
                
        except ImportError:
            # python-chess not available, use simple heuristic
            return _get_best_move_simple_heuristic(fen)
            
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "get_best_chess_move",
            "error": f"Error in fallback chess evaluation: {str(e)}"
        })

def _try_stockfish_online_api_v2(fen: str, depth: int = 15) -> str:
    """
    Try to get best move using Stockfish Online API v2 (https://stockfish.online/api/s/v2.php).
    Based on the official documentation. Adds debug output for troubleshooting.
    """
    try:
        # Use Stockfish Online API v2
        api_url = "https://stockfish.online/api/s/v2.php"
        params = {
            'fen': fen,
            'depth': depth
        }
        print(f"[DEBUG] Requesting Stockfish API: {api_url}")
        print(f"[DEBUG] Params: {params}")
        response = requests.get(api_url, params=params, timeout=15)
        print(f"[DEBUG] Status code: {response.status_code}")
        print(f"[DEBUG] Response text: {response.text}")
        if response.status_code == 200:
            data = response.json()
            # Check if request was successful
            if data.get('success') == True:
                bestmove = data.get('bestmove', '')
                if bestmove:
                    # Extract the actual move from the bestmove string
                    # Format: "bestmove b7b6 ponder f3e5" -> extract "b7b6"
                    move_parts = bestmove.split()
                    if len(move_parts) >= 2 and move_parts[0] == 'bestmove':
                        return move_parts[1]  # Return the actual move
                    else:
                        return bestmove  # Return full string if parsing fails
                else:
                    return json.dumps({
                        "type": "tool_response",
                        "tool_name": "get_best_chess_move",
                        "error": "Error: No bestmove in Stockfish API response",
                        "api_response": data
                    })
            else:
                error_msg = data.get('data', 'Unknown error')
                return json.dumps({
                    "type": "tool_response",
                    "tool_name": "get_best_chess_move",
                    "error": f"Error: Stockfish API failed - {error_msg}",
                    "api_response": data
                })
        return json.dumps({
            "type": "tool_response",
            "tool_name": "get_best_chess_move",
            "error": f"Error: Stockfish API returned status {response.status_code}",
            "response_text": response.text
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "get_best_chess_move",
            "error": f"Error accessing Stockfish Online API v2: {str(e)}"
        })

def _evaluate_moves_simple(board, legal_moves):
    """
    Simple move evaluation for when no chess engine is available.
    """
    try:
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            score = 0
            
            # Check if move captures a piece
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    # Piece values: Q=9, R=5, B=3, N=3, P=1
                    piece_values = {'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1}
                    score += piece_values.get(captured_piece.symbol().upper(), 1)
            
            # Check if move gives check
            board.push(move)
            if board.is_check():
                score += 2
            board.pop()
            
            # Prefer center moves for pawns
            if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol().upper() == 'P':
                center_files = ['d', 'e']
                if chr(ord('a') + move.to_square % 8) in center_files:
                    score += 1
            
            # Prefer developing moves (moving pieces from back rank)
            if move.from_square // 8 in [0, 7]:  # Back ranks
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
        
    except Exception as e:
        return None

def _get_best_move_simple_heuristic(fen: str) -> str:
    """
    Simple heuristic-based move selection when no chess engine is available.
    This analyzes the position and makes intelligent move decisions.
    """
    try:
        # Parse FEN to understand the position
        parts = fen.split()
        if len(parts) < 1:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "get_best_chess_move",
                "error": "Error: Invalid FEN format"
            })
        
        board_part = parts[0]
        side_to_move = parts[1] if len(parts) > 1 else 'w'
        ranks = board_part.split('/')
        
        # Convert FEN to a more analyzable format
        board = []
        for rank in ranks:
            row = []
            for char in rank:
                if char.isdigit():
                    row.extend([''] * int(char))
                else:
                    row.append(char)
            board.append(row)
        
        # Find all pieces for the side to move
        pieces = []
        for rank_idx, rank in enumerate(board):
            for file_idx, piece in enumerate(rank):
                if piece:
                    # Determine if piece belongs to side to move
                    is_white_piece = piece.isupper()
                    is_black_piece = piece.islower()
                    
                    if (side_to_move == 'w' and is_white_piece) or (side_to_move == 'b' and is_black_piece):
                        pieces.append({
                            'piece': piece.lower(),
                            'rank': rank_idx,
                            'file': file_idx,
                            'square': chr(ord('a') + file_idx) + str(8 - rank_idx)
                        })
        
        # Simple move selection based on piece values and position
        # Priority: Queen > Rook > Bishop > Knight > Pawn
        piece_values = {'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1}
        
        # Sort pieces by value (highest first)
        pieces.sort(key=lambda p: piece_values.get(p['piece'], 0), reverse=True)
        
        # For now, return a move from the highest value piece
        # This is a simplified approach - in reality you'd want to analyze legal moves
        if pieces:
            piece = pieces[0]
            # Create a simple move (this is just a placeholder)
            # In a real implementation, you'd generate legal moves for this piece
            from_square = piece['square']
            
            # Simple heuristic: try to move towards center or capture
            if piece['piece'] == 'p':  # Pawn
                # Move pawn forward
                if side_to_move == 'w':
                    to_rank = piece['rank'] - 1
                else:
                    to_rank = piece['rank'] + 1
                
                if 0 <= to_rank < 8:
                    to_square = chr(ord('a') + piece['file']) + str(8 - to_rank)
                    return from_square + to_square
            
            elif piece['piece'] == 'q':  # Queen
                # Try to move queen to center or capture
                center_squares = ['d4', 'e4', 'd5', 'e5']
                for center in center_squares:
                    if center != from_square:
                        return from_square + center
            
            elif piece['piece'] == 'r':  # Rook
                # Try to move rook to open file or rank
                return from_square + 'd' + str(8 - piece['rank'])
            
            elif piece['piece'] == 'b':  # Bishop
                # Try to move bishop to long diagonal
                return from_square + 'd4'
            
            elif piece['piece'] == 'n':  # Knight
                # Try to move knight towards center
                return from_square + 'd4'
            
            elif piece['piece'] == 'k':  # King
                # Try to castle or move king to safety
                return from_square + 'g1' if side_to_move == 'w' else from_square + 'g8'
        
        # Fallback: return a basic move
        return json.dumps({
            "type": "tool_response",
            "tool_name": "get_best_chess_move",
            "result": "e2e4" if side_to_move == 'w' else "e7e5"
        })
        
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "get_best_chess_move",
            "error": f"Error in simple heuristic: {str(e)}"
        })

# ========== FEN HELPER FUNCTIONS ==========

@tool
def get_best_chess_move(fen: str, original_input: str = None) -> str:
    """
    Get the best chess move candidates in coordinate notation based on a FEN representation using multiple chess evaluation sources.
    The result is a structured object containing:
      - The FEN string used for evaluation
      - The original input (if provided)
      - A list of candidate moves, each with its source and explanation
    The LLM should analyze the candidates and explanations to decide which move is best for the context.
    The FEN (Forsyth-Edwards Notation) describes the current chess position.
    Eg. rn1q1rk1/pp2b1pp/2p2n2/3p1pB1/3P4/1QP2N2/PP1N1PPP/R4RK1 b - - 1 11
    This tool tries several candidate sources (Lichess cloud eval, Stockfish Online API, local python-chess Stockfish, simple heuristics)

    Args:
        fen (str): The chess position in FEN (Forsyth-Edwards Notation) format.
        original_input (str, optional): The original chess problem or input details.

    Returns:
        str: JSON string with all move candidates and their explanations, for LLM reasoning.
    """
    result = _get_best_chess_move_internal(fen)
    # Attach original_input if provided
    if isinstance(result, dict):
        result["original_input"] = original_input
    return json.dumps({
        "type": "tool_response",
        "tool_name": "get_best_chess_move",
        "fen": result.get("fen"),
        "original_input": result.get("original_input"),
        "candidates": result.get("candidates", [])
    })

@tool
def solve_chess_position(image_path: str, player_turn: str, question: str = "") -> str:
    """
    Solve a chess position by analyzing the board image and finding the best move.
    This tool returns a structured object containing:
      - The extracted FEN (with explanation)
      - The original input details (image path, player turn, question)
      - A list of candidate moves (with explanations)
    The LLM should analyze the candidates and explanations to decide which move is best for the context.

    Args:
        image_path (str): The path to the chess board image file or base64-encoded image data.
        player_turn (str): The player with the next turn ("black" or "white").
        question (str): Optional question about the position (e.g., "guarantees a win").

    Returns:
        str: JSON string with all details and move candidates for LLM reasoning.
    """
    # Step 1: Get FEN from image
    fen_explanation = ""
    fen = None
    try:
        fen_result = _get_chess_board_fen_internal(image_path)
        if isinstance(fen_result, str) and fen_result.startswith("Error"):
            fen_explanation = fen_result
            fen = None
        else:
            fen = fen_result
            fen_explanation = "FEN extracted successfully from image."
    except Exception as e:
        fen_explanation = f"Error extracting FEN: {str(e)}"
        fen = None
    # Step 2: Get best move candidates (if FEN available)
    candidates = []
    if fen:
        best_move_result = _get_best_chess_move_internal(fen)
        if isinstance(best_move_result, dict):
            candidates = best_move_result.get('candidates', [])
        else:
            candidates = []
    return json.dumps({
        'type': 'tool_response',
        'tool_name': 'solve_chess_position',
        'fen': fen,
        'fen_explanation': fen_explanation,
        'original_input': {
            'image_path': image_path,
            'player_turn': player_turn,
            'question': question
        },
        'candidates': candidates
    })

# ========== FEN PROCESSING HELPERS ==========
def _add_fen_game_state(board_placement,
                    side_to_move,
                    castling="-",
                    en_passant="-",
                    halfmove_clock=0,
                    fullmove_number=1):
    """
    Appends standard game state information to a FEN board placement string.

    Args:
        board_placement (str): The board layout part of the FEN string
                            (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").
        side_to_move (str): The active color ('w' for White, 'b' for Black).
                            Case-insensitive, will be converted to lowercase.
        castling (str, optional): Castling availability string (e.g., "KQkq", "-").
                                Defaults to "-".
        en_passant (str, optional): En passant target square string (e.g., "e3", "-").
                                    Defaults to "-".
        halfmove_clock (int, optional): The number of halfmoves since the last
                                    capture or pawn advance. Defaults to 0.
        fullmove_number (int, optional): The number of the full move. Starts at 1
                                    and increments after Black's move. Defaults to 1.

    Returns:
        str: The complete FEN string including the game state,
            or an error message string if inputs are invalid.
    """
    # Validate side_to_move
    side_to_move_lower = str(side_to_move).lower()
    if side_to_move_lower not in ['w', 'b']:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "add_fen_game_state",
            "error": f"Error: side_to_move must be 'w' or 'b', received '{side_to_move}'"
        })

    # Validate clock values (should be non-negative integers, fullmove >= 1)
    try:
        halfmove_clock = int(halfmove_clock)
        fullmove_number = int(fullmove_number)
        if halfmove_clock < 0:
            raise ValueError("halfmove_clock cannot be negative.")
        if fullmove_number < 1:
            raise ValueError("fullmove_number must be 1 or greater.")
    except (ValueError, TypeError):
        return json.dumps({
            "type": "tool_response",
            "tool_name": "add_fen_game_state",
            "error": f"Error: halfmove_clock ('{halfmove_clock}') and "
                    f"fullmove_number ('{fullmove_number}') must be valid integers "
                    f"(non-negative and positive respectively)."
        })

    # Assemble the full FEN string using the validated/defaulted values
    # Note: castling and en_passant strings are used directly as passed or defaulted.
    # More complex validation could be added for them if needed.
    full_fen = (f"{board_placement} {side_to_move_lower} {castling} "
                f"{en_passant} {halfmove_clock} {fullmove_number}")

    return json.dumps({
        "type": "tool_response",
        "tool_name": "add_fen_game_state",
        "result": full_fen
    })

def _fen_normalize(fen: str, default_side='w'):
    """
    Normalize and validate a FEN string. Always return a best-effort valid FEN.
    - If only the board part is present, append default fields.
    - If FEN is valid, return as is.
    - If not valid, try to fix or return a clear error FEN.
    """
    fen = fen.strip()
    parts = fen.split()
    # If only board part, append defaults
    if len(parts) == 1 and parts[0].count('/') == 7:
        fen = f"{fen} {default_side} - - 0 1"
    # Validate using python-chess
    try:
        board = chess.Board(fen)
        return board.fen()
    except Exception as e:
        return f"8/8/8/8/8/8/8/8 w - - 0 1"  # Return an empty board as a fallback

def _get_chess_board_fen_internal(image_input: str) -> str:
    """
    Internal function to get the FEN representation from an image of a chess board.
    Uses the DerekLiu35-ImageToFen Hugging Face Space API.
    Args:
        image_input (str): Path to the chessboard image file or base64-encoded image data.
    Returns:
        str: The FEN string predicted by the recognizer, or an error message.
    """
    api_url = "https://DerekLiu35-ImageToFen.hf.space/api/predict"
    try:
        # Detect if input is a file path or base64 data
        if os.path.exists(image_input):
            with open(image_input, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            img_b64 = image_input
        payload = {"data": [img_b64]}
        response = requests.post(api_url, json=payload, timeout=60)
        if response.ok:
            result = response.json()
            data = result.get("data", [])
            if data:
                # FEN is usually the last string in the list
                fen_candidate = data[-1]
                if isinstance(fen_candidate, str) and fen_candidate.count('/') == 7:
                    return _fen_normalize(fen_candidate)
                # Fallback: search for a line with 7 slashes
                for item in data:
                    if isinstance(item, str) and item.count('/') == 7:
                        return _fen_normalize(item)
            return json.dumps({
                "type": "tool_response",
                "tool_name": "get_chess_board_fen",
                "error": f"Error: FEN not found in API response: {result}"
            })
        else:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "get_chess_board_fen",
                "error": f"Error: API call failed: {response.text}"
            })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "get_chess_board_fen",
            "error": f"Error running image-to-FEN API: {str(e)}"
        })
@tool
def get_chess_board_fen(image_path: str, player_turn: str) -> str:
    """
    Get the FEN representation from an image of a chess board.
    This tool uses computer vision to analyze a chess board image and convert it
    to FEN (Forsyth-Edwards Notation) format.
    Args:
        image_path (str): The path to the chess board image file.
        player_turn (str): The player with the next turn ("black" or "white").
    Returns:
        str: The FEN representation of the chess position, or error message.
    """
    fen = _get_chess_board_fen_internal(image_path)
    # If the result is a JSON error, pass it through
    try:
        import json
        data = json.loads(fen)
        if isinstance(data, dict) and 'error' in data:
            return fen
    except Exception:
        pass
    # Otherwise, return the normalized FEN in the required structure
    return json.dumps({
        "type": "tool_response",
        "tool_name": "get_chess_board_fen",
        "result": _fen_normalize(fen, default_side='b' if player_turn.lower().startswith('b') else 'w')
    })

@tool
def web_search_deep_research_exa_ai(instructions: str) -> str:
    """
    Search the web and site content using deep research tool.
    Ask a query and get a well-researched answer with references.
    Can provide FINAL ANSWER candidate.
    Ideal for research tasks on any topic that require fact searching.
    Can find answers and reference about science, scholars, sports, events, books, films, movies, mems, citations, etc.

    The tool researches a topic, verifies facts and outputs a structured answer.
    It deeply crawls websites to find the right answer, results and links.
    
    RESPONSE STRUCTURE:
    The tool returns a structured response with the following format:
    1. Task ID and Status
    2. Original Instructions
    3. Inferred Schema (JSON schema describing the response data structure)
    4. Data (JSON object containing the answer according to the schema)
    5. Citations (source references)
    
    SCHEMA INFERENCE:
    The tool automatically infers the appropriate schema based on your question.
    For example, a schema might include:
    - Person data: {"firstName", "lastName", "nationality", "year", etc.}
    - Event data: {"event", "date", "location", "participants", etc.}
    - Fact data: {"fact", "source", "context", etc.}
    
    DATA EXTRACTION:
    To extract the answer from the response:
    1. Look for the "Data" section in the response
    2. Parse the JSON object in the "Data" field  according to the schema
    3. Extract the relevant fields based on your question
    
    Args:
        instructions (str): Direct question or research instructions.

    Returns:
        str: The research result as a structured JSON string with schema, data, and citations, or an error message.
    """
    if not EXA_AVAILABLE:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "web_search_deep_research_exa_ai",
            "error": "Exa not available. Install with: pip install exa-py"
        })
    try:
        exa_api_key = os.environ.get("EXA_API_KEY")
        if not exa_api_key:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "web_search_deep_research_exa_ai",
                "error": "EXA_API_KEY not found in environment variables. Please set it in your .env file."
            })
        exa = Exa(exa_api_key)
        task_stub = exa.research.create_task(
            instructions=instructions,
            model="exa-research-pro",
            output_infer_schema = True
        )
        task = exa.research.poll_task(task_stub.id)
        return json.dumps({
            "type": "tool_response",
            "tool_name": "web_search_deep_research_exa_ai",
            "result": str(task)
        })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "web_search_deep_research_exa_ai",
            "error": f"Error in Exa research: {str(e)}"
        })

# ========== END OF TOOLS.PY ========== 
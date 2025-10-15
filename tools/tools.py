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

# Check if we're running on Hugging Face Spaces
HF_SPACES = os.environ.get('SPACE_ID') is not None
import cmath
import time
import re
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from typing import Any, Dict, List, Optional, Union, Tuple, Literal
from pydantic import BaseModel, Field, field_validator

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, Exception) as e:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    print(f"Warning: matplotlib not available: {e}")

# Try to import pytesseract for OCR
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

# Always import the tool decorator - it's essential
from langchain_core.tools import tool

# Expose Comindware Platform tools from all directories
# Templates tools
from .templates_tools.tool_list_attributes import list_attributes
from .templates_tools.tool_list_records import list_template_records

# Applications tools
from .applications_tools.tool_list_templates import list_templates
from .applications_tools.tool_list_applications import list_applications

# Templates tools
from .templates_tools.tools_record_template import edit_or_create_record_template

# Attributes tools - Text attributes
from .attributes_tools.tools_text_attribute import edit_or_create_text_attribute

# Attributes tools - Other attribute types
from .attributes_tools.tools_datetime_attribute import edit_or_create_date_time_attribute
from .attributes_tools.tools_decimal_attribute import edit_or_create_numeric_attribute
from .attributes_tools.tools_record_attribute import edit_or_create_record_attribute
from .attributes_tools.tools_image_attribute import edit_or_create_image_attribute
from .attributes_tools.tools_drawing_attribute import edit_or_create_drawing_attribute
from .attributes_tools.tools_document_attribute import edit_or_create_document_attribute
from .attributes_tools.tools_duration_attribute import edit_or_create_duration_attribute
from .attributes_tools.tools_account_attribute import edit_or_create_account_attribute
from .attributes_tools.tools_boolean_attribute import edit_or_create_boolean_attribute
from .attributes_tools.tools_role_attribute import edit_or_create_role_attribute
from .attributes_tools.tools_enum_attribute import edit_or_create_enum_attribute

# Attributes tools - Utility functions
from .attributes_tools.tool_delete_attribute import delete_attribute
from .attributes_tools.tool_archive_or_unarchive_attribute import archive_or_unarchive_attribute
from .attributes_tools.tool_get_attribute import get_attribute

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

# Try to import PyMuPDF for PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. Install with: pip install pymupdf")

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

# Google Gemini imports for video/audio understanding
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
        if HF_SPACES:
            return {"status": "error", "stderr": "Bash execution not available on Hugging Face Spaces"}
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
        if HF_SPACES:
            return {"status": "error", "stderr": "C code execution not available on Hugging Face Spaces"}
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
        if HF_SPACES:
            return {"status": "error", "stderr": "Java code execution not available on Hugging Face Spaces"}
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
def execute_code_multilang(code_reference: str, language: str = "python", agent=None) -> str:
    """Execute code in multiple languages (Python, Bash, SQL, C, Java) and return results.

    Args:
        code_reference (str): The source code to execute, filename, or URL to download code from.
        language (str): The language of the code. Supported: "python", "bash", "sql", "c", "java".
        agent: Agent instance for file resolution (injected automatically).

    Returns:
        A string summarizing the execution results (stdout, stderr, errors, plots, dataframes if any).
    """
    from .file_utils import FileUtils
    # Resolve code reference (code content, filename, or URL)
    code_content, detected_language = FileUtils.resolve_code_input(code_reference, agent)
    # Use detected language or provided language
    final_language = (detected_language or language).lower()
    supported_languages = ["python", "bash", "sql", "c", "java"]
    if final_language not in supported_languages:
        return FileUtils.create_tool_response(
            "execute_code_multilang", 
            error=f"❌ Unsupported language: {final_language}. Supported languages are: {', '.join(supported_languages)}"
        )

    result = interpreter_instance.execute_code(code_content, language=final_language)

    response = []

    if result["status"] == "success":
        response.append(f"✅ Code executed successfully in **{final_language.upper()}**")

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
        response.append(f"❌ Code execution failed in **{final_language.upper()}**")
        if result.get("stderr"):
            response.append(
                "\n**Error Log:**\n```\n" + result["stderr"].strip() + "\n```"
            )

    return FileUtils.create_tool_response("execute_code_multilang", result="\n".join(response))

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
        if not PYMUPDF_AVAILABLE:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "arxiv_search",
                "error": "PyMuPDF package not found, please install it with pip install pymupdf"
            })
        search_docs = ArxivLoader(query=input, load_max_docs=SEARCH_LIMIT).load()
        formatted_results = "\n\n---\n\n".join(
            [
                f'<Document title="{doc.metadata.get("Title", "Unknown")}" authors="{doc.metadata.get("Authors", "Unknown")}" published="{doc.metadata.get("Published", "Unknown")}"/>\n{doc.page_content}'
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
# ========== FILE/DATA TOOLS ==========
@tool
def read_text_based_file(file_reference: str, agent=None) -> str:
    """
    Read text-based files and return raw content as text.
    This is the general-purpose text file reader that handles most text-based formats.
    Use this tool for any text file unless there's a specialized tool available.
    Supported file types:
    - Text files: .txt, .md, .log, .rtf
    - Code files: .py, .js, .ts, .html, .htm, .css, .sql, .java, .cpp, .c, .php, .rb, .go
    - Configuration files: .ini, .cfg, .conf, .env, .properties, .yaml, .yml
    - Structured text: .json, .xml, .svg
    - Web files: .html, .htm, .xml
    - Documentation: .md, .rst, .tex
    - PDF files: .pdf (extracts text content and metadata)
    Note: For specialized analysis, use dedicated tools instead:
    - CSV files (.csv, .tsv): use analyze_csv_file
    - Excel files (.xlsx, .xls): use analyze_excel_file
    - Images: use analyze_image or extract_text_from_image
    The tool automatically:
    - Detects file encoding and handles multiple encodings (UTF-8, Latin-1, CP1252, ISO-8859-1)
    - Resolves filenames to full file paths via agent's file registry
    - Downloads files from URLs automatically
    - Provides file metadata (name, size, encoding) in results
    Args:
        file_reference (str): Original filename from user upload OR URL to download
        agent: Agent instance for file resolution (injected automatically)
    Returns:
        str: The raw text content of the file with metadata, or an error message if reading fails
    """
    from .file_utils import FileUtils
    # Resolve file reference (filename or URL) to full path
    file_path = FileUtils.resolve_file_reference(file_reference, agent)
    if not file_path:
        return FileUtils.create_tool_response("read_text_based_file", error=f"File not found: {file_reference}")
    # Get file info for validation
    file_info = FileUtils.get_file_info(file_path)
    if not file_info.exists:
        return FileUtils.create_tool_response("read_text_based_file", error=file_info.error)
    # Check if it's a PDF file and handle accordingly
    if file_info.extension == '.pdf':
        try:
            from .pdf_utils import PDFUtils
            if not PDFUtils.is_available():
                return FileUtils.create_tool_response("read_text_based_file", error="PyMuPDF not available. Install with: pip install pymupdf", file_info=file_info)
            # Extract text from PDF using LangChain and PyMuPDF4LLM
            pdf_result = PDFUtils.extract_text_from_pdf(file_path, use_markdown=True)
            if not pdf_result.success:
                return FileUtils.create_tool_response("read_text_based_file", error=pdf_result.error_message, file_info=file_info)
            # Show original reference in result
            display_name = file_reference if file_reference.startswith(('http://', 'https://', 'ftp://')) else file_reference
            size_str = FileUtils.format_file_size(file_info.size)
            # Format the response - just the content, no internal details
            content = pdf_result.text_content
            result_text = f"File: {display_name} ({size_str})\n\nContent:\n{content}"
            return FileUtils.create_tool_response("read_text_based_file", result=result_text, file_info=file_info)
        except Exception as e:
            return FileUtils.create_tool_response("read_text_based_file", error=f"Error processing PDF: {str(e)}", file_info=file_info)
    # Use modular file utilities with Pydantic validation for other text files
    result = FileUtils.read_text_file(file_path)
    if not result.success:
        return FileUtils.create_tool_response("read_text_based_file", error=result.error)
    # Format the result
    file_info = result.file_info
    content = result.content
    encoding = result.encoding
    size_str = FileUtils.format_file_size(file_info.size)
    # Show original reference in result
    display_name = file_reference if file_reference.startswith(('http://', 'https://', 'ftp://')) else file_reference
    if encoding != 'utf-8':
        result_text = f"File: {display_name} ({size_str}, {encoding} encoding)\n\nContent:\n{content}"
    else:
        result_text = f"File: {display_name} ({size_str})\n\nContent:\n{content}"
    return FileUtils.create_tool_response("read_text_based_file", result=result_text, file_info=file_info)

@tool
def extract_text_from_image(file_reference: str, agent=None) -> str:
    """
    Extract text from an image file using OCR (pytesseract) and return the extracted text.

    Args:
        file_reference (str): Original filename from user upload OR URL to download
        agent: Agent instance for file resolution (injected automatically)

    Returns:
        str: The extracted text, or an error message if extraction fails.
    """
    from .file_utils import FileUtils
    try:
        # Resolve file reference (filename or URL) to full path
        file_path = FileUtils.resolve_file_reference(file_reference, agent)
        if not file_path:
            return FileUtils.create_tool_response("extract_text_from_image", error=f"File not found: {file_reference}")
        image = Image.open(file_path)
        if PYTESSERACT_AVAILABLE:
            text = pytesseract.image_to_string(image)
        else:
            return FileUtils.create_tool_response(
                "extract_text_from_image", 
                error="OCR not available. Install with: pip install pytesseract"
            )
        return FileUtils.create_tool_response("extract_text_from_image", result=f"Extracted text from image:\n\n{text}")
    except Exception as e:
        return FileUtils.create_tool_response("extract_text_from_image", error=f"Error extracting text from image: {str(e)}")

@tool
def analyze_csv_file(file_reference: str, query: str, agent=None) -> str:
    """
    Analyze CSV files and return summary statistics and column information.
    This tool can process CSV files with various formats and encodings:
    - Standard CSV files: .csv
    - Tab-separated files: .tsv, .txt (with tab delimiters)
    - Comma-separated files with different encodings
    The tool automatically:
    - Detects delimiters and handles common CSV variations
    - Resolves filenames to full file paths via agent's file registry
    - Downloads files from URLs automatically
    - Provides comprehensive analysis including data types, statistics, and column information
    Args:
        file_reference (str): Original filename from user upload OR URL to download
        query (str): A question or description of the analysis to perform (currently unused)
        agent: Agent instance for file resolution (injected automatically)

    Returns:
        str: Summary statistics and column information, or an error message if analysis fails.
    """
    from .file_utils import FileUtils
    # Resolve file reference (filename or URL) to full path
    file_path = FileUtils.resolve_file_reference(file_reference, agent)
    if not file_path:
        return FileUtils.create_tool_response("analyze_csv_file", error=f"File not found: {file_reference}")
    # Check file exists using utilities with Pydantic validation
    file_info = FileUtils.get_file_info(file_path)
    if not file_info.exists:
        return FileUtils.create_tool_response("analyze_csv_file", error=file_info.error)
    try:
        df = pd.read_csv(file_path)
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"File: {file_info.name} ({FileUtils.format_file_size(file_info.size)})\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        result += "Summary statistics:\n"
        result += str(df.describe())
        return FileUtils.create_tool_response("analyze_csv_file", result=result, file_info=file_info)
    except Exception as e:
        return FileUtils.create_tool_response("analyze_csv_file", error=f"Error analyzing CSV file: {str(e)}")

@tool
def analyze_excel_file(file_reference: str, query: str, agent=None) -> str:
    """
    Analyze Excel files and return summary statistics and column information.
    This tool can process Excel files in various formats:
    - Excel files: .xlsx, .xls
    - Excel workbooks with multiple sheets
    - Excel files with different encodings and formats
    The tool automatically:
    - Detects sheet structure and provides comprehensive analysis
    - Resolves filenames to full file paths via agent's file registry
    - Downloads files from URLs automatically
    - Provides data types, statistics, column information, and sheet details
    Args:
        file_reference (str): Original filename from user upload OR URL to download
        query (str): A question or description of the analysis to perform (currently unused)
        agent: Agent instance for file resolution (injected automatically)

    Returns:
        str: Summary statistics and column information, or an error message if analysis fails.
    """
    from .file_utils import FileUtils
    # Resolve file reference (filename or URL) to full path
    file_path = FileUtils.resolve_file_reference(file_reference, agent)
    if not file_path:
        return FileUtils.create_tool_response("analyze_excel_file", error=f"File not found: {file_reference}")
    # Check file exists using utilities with Pydantic validation
    file_info = FileUtils.get_file_info(file_path)
    if not file_info.exists:
        return FileUtils.create_tool_response("analyze_excel_file", error=file_info.error)
    try:
        df = pd.read_excel(file_path)
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"File: {file_info.name} ({FileUtils.format_file_size(file_info.size)})\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        result += "Summary statistics:\n"
        result += str(df.describe())
        return FileUtils.create_tool_response("analyze_excel_file", result=result, file_info=file_info)
    except Exception as e:
        # Enhanced error reporting: print columns and head if possible
        try:
            df = pd.read_excel(file_path)
            columns = list(df.columns)
            head = df.head().to_dict('records')
            error_details = f"Error analyzing Excel file: {str(e)}\nColumns: {columns}\nHead: {head}"
        except Exception as inner_e:
            error_details = f"Error analyzing Excel file: {str(e)}\nAdditionally, failed to read columns/head: {str(inner_e)}"
        return FileUtils.create_tool_response("analyze_excel_file", error=error_details)

# ========== IMAGE ANALYSIS/GENERATION TOOLS ==========
@tool
def analyze_image(file_reference: str, agent=None) -> str:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview) from a file reference.

    The tool automatically:
    - Resolves filenames to full file paths via agent's file registry
    - Downloads files from URLs automatically
    - Provides comprehensive image analysis including dimensions, color analysis, and thumbnails

    Args:
        file_reference (str): Original filename from user upload OR URL to download
        agent: Agent instance for file resolution (injected automatically)

    Returns:
        str: JSON string with analysis results including dimensions, mode, color_analysis, and thumbnail.
    """
    from .file_utils import FileUtils
    try:
        # Resolve file reference (filename or URL) to full path
        file_path = FileUtils.resolve_file_reference(file_reference, agent)
        if not file_path:
            return FileUtils.create_tool_response("analyze_image", error=f"File not found: {file_reference}")
        # Open image from file path
        img = Image.open(file_path)
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
        return FileUtils.create_tool_response("analyze_image", result=json.dumps(result))
    except Exception as e:
        return FileUtils.create_tool_response("analyze_image", error=str(e))

class TransformImageParams(BaseModel):
    width: Optional[int] = Field(None, description="New width for resize operation")
    height: Optional[int] = Field(None, description="New height for resize operation")
    angle: Optional[int] = Field(None, description="Rotation angle in degrees")
    direction: Optional[Literal["horizontal", "vertical"]] = Field(None, description="Flip direction")
    radius: Optional[float] = Field(None, description="Blur radius")
    factor: Optional[float] = Field(None, description="Enhancement factor for brightness/contrast")

@tool(args_schema=TransformImageParams)
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

class DrawOnImageParams(BaseModel):
    text: Optional[str] = Field(None, description="Text to draw")
    position: Optional[List[int]] = Field(None, description="Text position [x, y]")
    color: Optional[str] = Field(None, description="Color name (e.g., 'red', 'blue') or RGB string (e.g., '255,0,0')")
    size: Optional[int] = Field(None, description="Font size for text")
    coords: Optional[List[int]] = Field(None, description="Rectangle coordinates [x1, y1, x2, y2]")
    center: Optional[List[int]] = Field(None, description="Circle center [x, y]")
    radius: Optional[int] = Field(None, description="Circle radius")
    start: Optional[List[int]] = Field(None, description="Line start [x, y]")
    end: Optional[List[int]] = Field(None, description="Line end [x, y]")
    width: Optional[int] = Field(None, description="Stroke width")

@tool(args_schema=DrawOnImageParams)
def draw_on_image(image_base64: str, drawing_type: str, params: DrawOnImageParams) -> str:
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
        def parse_color(color_str):
            """Parse color string to RGB tuple or color name"""
            if not color_str:
                return "black"
            # Check if it's RGB values as comma-separated string
            if "," in color_str and color_str.replace(",", "").replace(" ", "").isdigit():
                try:
                    rgb_values = [int(x.strip()) for x in color_str.split(",")]
                    if len(rgb_values) == 3 and all(0 <= v <= 255 for v in rgb_values):
                        return tuple(rgb_values)
                except ValueError:
                    pass
            # Return as color name
            return color_str

        if drawing_type == "text":
            text = params.text or ""
            position = params.position or [10, 10]
            color = parse_color(params.color) or "black"
            size = params.size or 20
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except:
                font = ImageFont.load_default()
            draw.text(tuple(position), text, fill=color, font=font)
        elif drawing_type == "rectangle":
            coords = params.coords or [10, 10, 100, 100]
            color = parse_color(params.color) or "red"
            width = params.width or 2
            draw.rectangle(coords, outline=color, width=width)
        elif drawing_type == "circle":
            center = params.center or [50, 50]
            radius = params.radius or 30
            color = parse_color(params.color) or "blue"
            width = params.width or 2
            bbox = [center[0] - radius, center[1] - radius, 
                   center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, outline=color, width=width)
        elif drawing_type == "line":
            start = params.start or [10, 10]
            end = params.end or [100, 100]
            color = parse_color(params.color) or "green"
            width = params.width or 2
            draw.line([tuple(start), tuple(end)], fill=color, width=width)
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

class GenerateSimpleImageParams(BaseModel):
    image_type: str = Field(..., description="Type of image to generate: 'solid', 'gradient', 'checkerboard', 'noise'")
    width: int = Field(500, description="Width of the generated image")
    height: int = Field(500, description="Height of the generated image")
    color: Optional[str] = Field(None, description="Solid color for 'solid' type (e.g., 'red', 'blue') or RGB string (e.g., '255,0,0')")
    start_color: Optional[List[int]] = Field(None, description="Gradient start color [r, g, b]")
    end_color: Optional[List[int]] = Field(None, description="Gradient end color [r, g, b]")
    direction: Optional[Literal["horizontal", "vertical"]] = Field(None, description="Gradient direction")
    square_size: Optional[int] = Field(None, description="Square size for checkerboard")
    color1: Optional[str] = Field(None, description="First color for checkerboard")
    color2: Optional[str] = Field(None, description="Second color for checkerboard")

@tool(args_schema=GenerateSimpleImageParams)
def generate_simple_image(image_type: str, width: int = 500, height: int = 500, 
                         color: Optional[str] = None, start_color: Optional[List[int]] = None,
                         end_color: Optional[List[int]] = None, direction: Optional[str] = None,
                         square_size: Optional[int] = None, color1: Optional[str] = None,
                         color2: Optional[str] = None) -> str:
    """
    Generate simple images like gradients, solid colors, checkerboard, or noise patterns.

    Args:
        image_type (str): The type of image to generate.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        color (str, optional): Solid color for 'solid' type or RGB string.
        start_color (List[int], optional): Gradient start color [r, g, b].
        end_color (List[int], optional): Gradient end color [r, g, b].
        direction (str, optional): Gradient direction ('horizontal' or 'vertical').
        square_size (int, optional): Square size for checkerboard.
        color2 (str, optional): Second color for checkerboard.

    Returns:
        str: JSON string with the generated image as base64 or error message.
    """
    try:
        if image_type == "solid":
            color_str = color or "255,255,255"
            # Parse color string to RGB tuple
            if "," in color_str and color_str.replace(",", "").replace(" ", "").isdigit():
                try:
                    rgb_values = [int(x.strip()) for x in color_str.split(",")]
                    if len(rgb_values) == 3 and all(0 <= v <= 255 for v in rgb_values):
                        color = tuple(rgb_values)
                    else:
                        color = (255, 255, 255)
                except ValueError:
                    color = (255, 255, 255)
            else:
                # Try as color name, fallback to white
                try:
                    color = color_str
                except:
                    color = (255, 255, 255)
            img = Image.new("RGB", (width, height), color)
        elif image_type == "gradient":
            start_color = start_color or [255, 0, 0]
            end_color = end_color or [0, 0, 255]
            direction = direction or "horizontal"
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
            square_size = square_size or 50
            color1 = color1 or "white"  # Use provided color1 or default to white
            color2 = color2 or "black"
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

class CombineImagesParams(BaseModel):
    spacing: Optional[int] = Field(None, description="Spacing between images in pixels")
    background_color: Optional[str] = Field(None, description="Background color for collage (e.g., 'white', 'black') or RGB string (e.g., '255,255,255')")
    blend_mode: Optional[str] = Field(None, description="Blend mode for blending operations")
    opacity: Optional[float] = Field(None, description="Opacity for overlay operations (0.0-1.0)")

@tool(args_schema=CombineImagesParams)
def combine_images(images_base64: List[str], operation: str, 
                  params: Optional[CombineImagesParams] = None) -> str:
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
        if params is None:
            params = CombineImagesParams()
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
            direction = params.direction or "horizontal"
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
def understand_video(file_reference: str, prompt: str, system_prompt: str = None, agent=None, 
                     start_time: str = None, end_time: str = None, fps: float = None) -> str:
    """
    Analyze a video using Google Gemini's video understanding capabilities.
    This tool can understand video content, extract information, answer questions,
    and provide transcriptions with timestamps. Supports video clipping and custom frame rates.
    Supports four input methods:
    1. Uploaded video files - File size >20MB
    2. Direct video URLs - File size >20MB
    3. YouTube URLs - No size limit
    4. Inline video data - For small videos <20MB
    Advanced features:
    - Video clipping: Specify start_time and end_time in MM:SS format (e.g., "02:30", "03:29")
    - Custom frame rate: Set fps for different sampling rates (default: 1 FPS)
    - Timestamp references: Use MM:SS format in prompts for specific video segments
    Args:
        file_reference (str): Original filename from user upload OR direct video URL 
                             OR YouTube URL OR base64 encoded video data (<20MB)
        prompt (str): A question or request regarding the video content
                        When referring to specific moments in a video within your prompt,
                        use the MM:SS format (e.g., "01:15" for 1 minute and 15 seconds).
        system_prompt (str, optional): System instruction
        agent: Agent instance for file resolution (injected automatically)
        start_time (str, optional): Start time for video clipping in MM:SS format (e.g., "02:30")
        end_time (str, optional): End time for video clipping in MM:SS format (e.g., "03:29")
        fps (float, optional): Custom frame rate for video processing (default: 1 FPS).
                               You might want to set low FPS (< 1) for long videos.
                               This is especially useful for mostly static videos (e.g. lectures). 
                               If you want to capture more details in rapidly changing visuals, 
                               consider setting a higher FPS value.
    Returns:
        str: Analysis of the video content based on the prompt, or error message
    """
    from .file_utils import FileUtils
    def create_video_metadata():
        """Create video metadata for clipping and frame rate if specified."""
        def time_to_seconds(time_str):
            """Convert MM:SS or raw seconds to API-required seconds format with 's' suffix.
            Examples:
                "02:30" -> "150s"
                "1:15" -> "75s"
                "1250" -> "1250s"
                "1250s" -> "1250s"
            """
            if not time_str:
                return None
            # If already has 's' suffix, return as-is
            if time_str.endswith('s'):
                return time_str
            # Check if it's MM:SS format
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = parts
                    total_seconds = int(minutes) * 60 + int(seconds)
                    return f"{total_seconds}s"
            # Assume it's already in seconds, add 's' suffix
            return f"{time_str}s"
        metadata = {}
        if start_time:
            metadata['start_offset'] = time_to_seconds(start_time)
        if end_time:
            metadata['end_offset'] = time_to_seconds(end_time)
        if fps is not None:
            metadata['fps'] = fps
        return metadata if metadata else None
    try:
        client = _get_gemini_client()
        if not client:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "understand_video",
                "error": "Gemini client not available. Check GEMINI_KEY environment variable."
            })
        # Create video metadata if any advanced features are specified
        video_metadata = create_video_metadata()
        # Determine input type and handle accordingly
        video_part = None
        # Check if it's a YouTube URL (special handling)
        if file_reference.startswith(('https://www.youtube.com/', 'https://youtube.com/', 
                                     'https://youtu.be/', 'http://www.youtube.com/',
                                     'http://youtube.com/', 'http://youtu.be/')):
            # YouTube URL - pass directly to Gemini with optional metadata
            if video_metadata:
                video_part = types.Part(
                    file_data=types.FileData(file_uri=file_reference),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
            else:
                video_part = types.Part(file_data=types.FileData(file_uri=file_reference))
        else:
            # Try to resolve as file reference (uploaded file or regular URL)
            resolved_path = FileUtils.resolve_file_reference(file_reference, agent)
            if resolved_path:
                # It's a file (uploaded or downloaded from URL)
                try:
                    uploaded_file = client.files.upload(file=resolved_path)
                    if video_metadata:
                        video_part = types.Part(
                            file_data=types.FileData(file_uri=uploaded_file.uri),
                            video_metadata=types.VideoMetadata(**video_metadata)
                        )
                    else:
                        video_part = types.Part(file_data=types.FileData(file_uri=uploaded_file.uri))
                except Exception as upload_error:
                    return json.dumps({
                        "type": "tool_response",
                        "tool_name": "understand_video",
                        "error": f"Error uploading video file to Gemini: {str(upload_error)}"
                    })
            else:
                # Try inline video data for small files (<20MB)
                try:
                    # Decode base64 and use inline data (not temporary file)
                    video_data = base64.b64decode(file_reference)
                    # Check size limit (20MB = 20 * 1024 * 1024 bytes)
                    if len(video_data) > 20 * 1024 * 1024:
                        return json.dumps({
                            "type": "tool_response",
                            "tool_name": "understand_video",
                            "error": "Video data too large for inline processing (>20MB). Please use file upload or URL instead."
                        })
                    # Use inline data for small videos with optional metadata
                    if video_metadata:
                        video_part = types.Part(
                            inline_data=types.Blob(
                                data=video_data,
                                mime_type='video/mp4'  # Default to mp4, could be detected from file extension
                            ),
                            video_metadata=types.VideoMetadata(**video_metadata)
                        )
                    else:
                        video_part = types.Part(
                            inline_data=types.Blob(
                                data=video_data,
                                mime_type='video/mp4'  # Default to mp4, could be detected from file extension
                            )
                        )
                except Exception as decode_error:
                    return json.dumps({
                        "type": "tool_response",
                        "tool_name": "understand_video",
                        "error": f"Error processing video data: {str(decode_error)}. Expected base64 encoded video data (<20MB), valid file path, YouTube URL, or direct video URL."
                    })
        # Don't embed system_prompt in user prompt - use API parameter instead
        enhanced_prompt = prompt
        # Generate content using the video
        contents = types.Content(
            parts=[
                video_part,
                types.Part(text=enhanced_prompt)
            ]
        )
        # Create config with system_instruction if provided
        config = None
        if system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config
            )
            return json.dumps({
                "type": "tool_response",
                "tool_name": "understand_video",
                "result": response.text
            })
        except Exception as e:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "understand_video",
                "error": f"Error in video understanding request: {str(e)}"
            })
    except Exception as e:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "understand_video",
            "error": f"Error understanding video: {str(e)}"
        })

@tool
def understand_audio(file_reference: str, prompt: str, system_prompt: str = None, agent=None, 
                     start_time: str = None, end_time: str = None) -> str:
    """
    Analyze an audio file using Google Gemini's audio understanding capabilities.
    This tool can transcribe audio, understand spoken content, and answer questions
    about the audio content. Supports timestamp references in prompts (MM:SS format).
    The audio file is uploaded to Gemini and then analyzed with the provided prompt.
    Args:
        file_reference (str): Original filename from user upload OR URL to download OR base64 encoded audio data.
        prompt (str): A question or request regarding the audio content.
        system_prompt (str, optional): System instruction.
        agent: Agent instance for file resolution (injected automatically)
        start_time (str, optional): Start time reference in MM:SS format (e.g., "02:30")
        end_time (str, optional): End time reference in MM:SS format (e.g., "03:29")
    Returns:
        str: Analysis of the audio content based on the prompt, or error message.
    """
    from .file_utils import FileUtils
    try:
        client = _get_gemini_client()
        if not client:
            return json.dumps({
                "type": "tool_response",
                "tool_name": "understand_audio",
                "error": "Gemini client not available. Check GEMINI_KEY environment variable."
            })

        # First try to resolve as file reference (uploaded file or URL)
        resolved_path = FileUtils.resolve_file_reference(file_reference, agent)

        if resolved_path:
            # It's a file (uploaded or downloaded from URL)
            try:
                mp3_file = client.files.upload(file=resolved_path)
            except Exception as upload_error:
                return json.dumps({
                    "type": "tool_response",
                    "tool_name": "understand_audio",
                    "error": f"Error uploading audio file to Gemini: {str(upload_error)}"
                })
        else:
            # Check if it looks like a URL that failed to download
            if file_reference.startswith(('http://', 'https://', 'ftp://')):
                return json.dumps({
                    "type": "tool_response",
                    "tool_name": "understand_audio",
                    "error": f"Failed to download audio from URL: {file_reference}. Please check the URL is accessible and try again."
                })

            # Try base64 fallback
            try:
                # Decode base64 and create temporary file
                audio_data = base64.b64decode(file_reference)
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
                    "error": f"Error processing audio data: {str(decode_error)}. Expected base64 encoded audio data, valid file path, or URL."
                })
        # Create enhanced prompt with timestamp references if provided
        timestamp_instruction = ""
        if start_time and end_time:
            timestamp_instruction = f" Focus on the audio segment from {start_time} to {end_time}."
        elif start_time:
            timestamp_instruction = f" Focus on the audio segment starting from {start_time}."
        elif end_time:
            timestamp_instruction = f" Focus on the audio segment up to {end_time}."
        # Build prompt with timestamp instructions only
        enhanced_prompt = f"{prompt}\n\n{timestamp_instruction}"
        contents = [enhanced_prompt, mp3_file]
        # Create config with system_instruction if provided
        config = None
        if system_prompt:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config
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

# ========== PYDANTIC SCHEMAS ==========

class SubmitAnswerSchema(BaseModel):
    """
    Schema for submitting final answers with structured metadata.
    Use this when ready to provide a final answer and the analysis is complete.
    """
    answer: str = Field(
        description="The final answer to the user's question",
        min_length=1
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level from 0.0 to 1.0 (default: 1.0)"
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="List of sources or tools used to generate this answer"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the reasoning process"
    )

    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v

class SubmitIntermediateStepSchema(BaseModel):
    """
    Schema for submitting intermediate reasoning steps or progress updates.
    Use this to document intermediate steps in your reasoning process, 
    progress updates, or partial findings before reaching a final conclusion.
    """
    step_name: str = Field(
        description="Short name/identifier for this step (e.g., 'data_analysis', 'search_results')",
        min_length=1,
        max_length=100
    )
    description: str = Field(
        description="Detailed description of what was accomplished in this step",
        min_length=1
    )
    status: Literal["in_progress", "completed", "failed", "blocked"] = Field(
        default="in_progress",
        description="Current status of this step"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary containing relevant data, findings, or results from this step"
    )
    next_steps: Optional[List[str]] = Field(
        default=None,
        description="Optional list of planned next steps or actions"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence level from 0.0 to 1.0 for this step's results"
    )
    issues: Optional[List[str]] = Field(
        default=None,
        description="Optional list of issues, concerns, or limitations encountered"
    )

    @field_validator('next_steps', 'issues')
    @classmethod
    def validate_lists(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v

class SubmitAnswerResult(BaseModel):
    """
    Structured result model for submit_answer operations.
    This model standardizes the response format for final answer submissions,
    providing consistent success/error handling and response structure.
    """
    success: bool
    status_code: int = Field(default=200)
    raw_response: dict | str | None = Field(default=None)
    error: Optional[str] = Field(default=None)

class SubmitIntermediateStepResult(BaseModel):
    """
    Structured result model for submit_intermediate_step operations.
    This model standardizes the response format for intermediate step submissions,
    providing consistent success/error handling and response structure.
    """
    success: bool
    status_code: int = Field(default=200)
    raw_response: dict | str | None = Field(default=None)
    error: Optional[str] = Field(default=None)

# ========== TOOL FUNCTIONS ==========

@tool("submit_answer", return_direct=False, args_schema=SubmitAnswerSchema)
def submit_answer(answer: str, confidence: float = 1.0, sources: List[str] = None, reasoning: str = None) -> Dict[str, Any]:
    """
    Submit a final answer using Schema-Guided Reasoning (SGR).
    This tool forces the LLM to explicitly state its conclusion rather than leaving it implicit.
    It preserves structured metadata while ensuring clean integration with the streaming pipeline.
    Use this tool when ready to provide an answer for the current question. This can be:
    - A final answer after completing all reasoning steps
    - An answer to a sub-question in a multi-turn conversation
    - A response that concludes the current analysis phase
    This tool preserves context across multiple conversation turns.
    Returns:
        dict: Structured response with answer and metadata
    """
    try:
        # Create structured response that's easy to extract
        result = {
            "success": True,
            "answer": answer,
            "confidence": confidence,
            "sources": sources or [],
            "reasoning": reasoning or "",
            "timestamp": time.time(),
            "type": "final_answer"
        }
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Error submitting answer: {str(e)}",
            "type": "error"
        }

@tool("submit_intermediate_step", return_direct=False, args_schema=SubmitIntermediateStepSchema)
def submit_intermediate_step(step_name: str, description: str, status: str = "in_progress", 
                           data: Dict[str, Any] = None, next_steps: List[str] = None, 
                           confidence: float = None, issues: List[str] = None) -> Dict[str, Any]:
    """
    Submit an intermediate reasoning step using Schema-Guided Reasoning (SGR).
    Use this tool to document intermediate steps in your reasoning process, 
    progress updates, or partial findings before reaching a final conclusion.
    This tool helps track the agent's thought process and enables better debugging.
    It helps guide structured thinking and makes the reasoning process transparent and debuggable.
    Returns:
        dict: Structured response with step details and metadata
    """
    try:
        # Create structured response that's easy to extract
        result = {
            "success": True,
            "step_name": step_name,
            "description": description,
            "status": status,
            "data": data or {},
            "next_steps": next_steps or [],
            "confidence": confidence,
            "issues": issues or [],
            "timestamp": time.time(),
            "type": "intermediate_step"
        }
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Error submitting step: {str(e)}",
            "type": "error"
        }
# ========== END OF TOOLS.PY ==========

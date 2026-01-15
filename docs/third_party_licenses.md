# Library Licenses for CMW Platform Agent

## Project License

The **CMW Platform Agent** project itself is licensed under the **MIT License**. [1](#0-0) 

## Internal Libraries

All internal modules developed as part of this project are under the **MIT License**:
- `agent_ng/` - Core agent implementation
- `cmw_open_api/` - CMW Platform API schemas
- `tools/` - Tool implementations

## External Dependencies

Based on the project's requirements files [2](#0-1) , here are the external libraries and their licenses:

### Core Framework Dependencies

| Library | Version | License |
|---------|---------|---------|
| **gradio** | >=4.0.0 | Apache 2.0 |
| **langchain** | 0.3.27 | MIT |
| **langchain-core** | 0.3.79 | MIT |
| **langchain-community** | 0.3.31 | MIT |

### LLM Provider Integrations

| Library | Version | License |
|---------|---------|---------|
| **openai** | 2.7.1 | Apache 2.0 |
| **langchain-openai** | 0.3.35 | MIT |
| **google-ai-generativelanguage** | 0.6.18 | Apache 2.0 |
| **google-api-core** | 2.25.1 | Apache 2.0 |
| **google-auth** | 2.40.3 | Apache 2.0 |
| **google-genai** | 1.36.0 | Apache 2.0 |
| **googleapis-common-protos** | 1.70.0 | Apache 2.0 |
| **langchain-google-genai** | 2.1.10 | MIT |
| **groq** | 0.33.0 | Apache 2.0 |
| **langchain-groq** | 0.3.8 | MIT |
| **mistralai** | 1.9.11 | Apache 2.0 |
| **langchain-mistralai** | 0.2.12 | MIT |
| **huggingface-hub** | 1.1.2 | Apache 2.0 |
| **langchain-huggingface** | 0.3.1 | MIT |
| **langchain-gigachat** | 0.3.12 | MIT |
| **gigachat** | 0.1.42.post2 | MIT |
| **langchain-tavily** | 0.2.13 | MIT |

### Data Processing & Analysis

| Library | Version | License |
|---------|---------|---------|
| **numpy** | 2.3.4 | BSD 3-Clause |
| **pandas** | 2.3.3 | BSD 3-Clause |
| **scikit-learn** | 1.7.2 | BSD 3-Clause |
| **pillow** | 12.0.0 | HPND (Historical Permission Notice and Disclaimer) |
| **matplotlib** | 3.10.7 | PSF (Python Software Foundation) |
| **pytesseract** | 0.3.13 | Apache 2.0 |

### Web & API Integrations

| Library | Version | License |
|---------|---------|---------|
| **requests** | 2.32.5 | Apache 2.0 |
| **python-dotenv** | 1.2.1 | BSD 3-Clause |
| **PyYAML** | 6.0.3 | MIT |
| **beautifulsoup4** | 4.14.2 | MIT |

### Document Processing

| Library | Version | License |
|---------|---------|---------|
| **pymupdf4llm** | 0.1.7 | AGPL-3.0 |
| **openpyxl** | >=3.1.0 | MIT |
| **Markdown** | 3.10 | BSD 3-Clause |

### Observability & Monitoring

| Library | Version | License |
|---------|---------|---------|
| **langsmith** | 0.4.41 | MIT |
| **langfuse** | 3.9.0 | MIT |

### Token Counting & Utilities

| Library | Version | License |
|---------|---------|---------|
| **tiktoken** | 0.12.0 | MIT |
| **pydantic** | 2.12.4 | MIT |

### Search & Research Tools

| Library | Version | License |
|---------|---------|---------|
| **wikipedia** | 1.4.0 | MIT |
| **arxiv** | 2.3.0 | MIT |

### Database & Storage (Optional)

| Library | Version | License |
|---------|---------|---------|
| **supabase** | >=2.0.0 | MIT |
| **pgvector** | >=0.2.0 | PostgreSQL License |

### Development & Testing

| Library | Version | License |
|---------|---------|---------|
| **pytest** | >=7.0.0 | MIT |
| **pytest-asyncio** | >=0.21.0 | Apache 2.0 |
| **ruff** | >=0.1.0 | MIT |
| **black** | >=23.0.0 | MIT |
| **flake8** | >=6.0.0 | MIT |

### Optional Advanced Features

| Library | Version | License |
|---------|---------|---------|
| **asyncio-mqtt** | >=0.11.0 | BSD 3-Clause |

## Notes

1. **Complete dependency list**: The full list of dependencies can be found in both `requirements.txt` [2](#0-1)  and `requirements_complete.txt` [3](#0-2)  files.

2. **License compatibility**: Most libraries use permissive licenses (MIT, Apache 2.0, BSD). However, **pymupdf4llm** uses AGPL-3.0, which is a copyleft license that may require special consideration if you plan to distribute the software.

3. **Internal modules**: All code in the `agent_ng/`, `cmw_open_api/`, and `tools/` directories is part of the project and covered under the MIT License specified in the README.

4. **Optional dependencies**: Some libraries in `requirements_complete.txt` are marked as optional and may not be required for all use cases.

### Citations

**File:** README.md (L11-11)
```markdown
license: mit
```

**File:** requirements.txt (L1-127)
```text
# CMW Platform Agent - Requirements for Hugging Face Spaces
# ======================================
# Only libraries actually used in agent_ng/ and tools/ directories for Hugging Face Spaces

# =============================================================================
# CORE FRAMEWORK DEPENDENCIES
# =============================================================================

# Gradio - Modern web UI framework
# Not needed in Huging Face, installed from README.md
gradio

# LangChain Ecosystem - Core AI framework
langchain==0.3.27
langchain-core==0.3.79
langchain-community==0.3.31

# =============================================================================
# LLM PROVIDER INTEGRATIONS
# =============================================================================

# OpenAI
openai==2.7.1
langchain-openai==0.3.35

# Google Gemini
google-ai-generativelanguage==0.6.18
google-api-core==2.25.1
google-auth==2.40.3
google-genai==1.36.0
googleapis-common-protos==1.70.0
langchain-google-genai==2.1.10

# Groq
groq==0.33.0
langchain-groq==0.3.8

# Mistral
mistralai==1.9.11
langchain-mistralai==0.2.12

# HuggingFace
huggingface-hub==1.1.2
langchain-huggingface==0.3.1

# GigaChat
langchain-gigachat==0.3.12
gigachat==0.1.42.post2

# Search Tools
langchain-tavily==0.2.13

# =============================================================================
# DATA PROCESSING & ANALYSIS
# =============================================================================

# Core Data Science
numpy==2.3.4
pandas==2.3.3
scikit-learn==1.7.2

# Image Processing
pillow==12.0.0

# Optional: Advanced Data Processing
matplotlib==3.10.7
pytesseract==0.3.13

# =============================================================================
# WEB & API INTEGRATIONS
# =============================================================================

# HTTP Requests
requests==2.32.5

# Configuration
python-dotenv==1.2.1

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

# PDF Processing - Best library for LLM processing
pymupdf4llm==0.1.7


# =============================================================================
# TEXT PROCESSING (used in chat_tab.py)
# =============================================================================

# Markdown Processing
Markdown==3.10

# =============================================================================

# Plotting
matplotlib==3.10.7
# OCR for image text extraction
pytesseract==0.3.13

# =============================================================================
# OBSERVABILITY
# =============================================================================

# LangSmith - Primary observability
langsmith==0.4.41

# Langfuse - Alternative observability
langfuse==3.9.0

# =============================================================================
# TOKEN COUNTING & UTILITIES
# =============================================================================

# Token Counting
tiktoken==0.12.0

# Data Validation
pydantic==2.12.4
# =============================================================================
# SEARCH & RESEARCH TOOLS
# =============================================================================

# Wikipedia
wikipedia==1.4.0

# ArXiv
```

**File:** requirements_complete.txt (L1-153)
```text
# CMW Platform Agent - Perfect Requirements
# ==========================================
# Comprehensive dependencies for agent_ng/, resources/, and tools/ directories
# Optimized for production deployment with development tools

# =============================================================================
# CORE FRAMEWORK DEPENDENCIES
# =============================================================================

# Gradio - Modern web UI framework
gradio>=4.0.0

# LangChain Ecosystem - Core AI framework
langchain==0.3.27
langchain-core==0.3.79
langchain-community==0.3.31

# =============================================================================
# LLM PROVIDER INTEGRATIONS
# =============================================================================

# OpenAI
openai==2.7.1
langchain-openai==0.3.35

# Google Gemini
google-genai==1.36.0
#google-generativeai>=0.8.5
langchain-google-genai==2.1.10

# Groq
groq==0.33.0
langchain-groq==0.3.8

# Mistral
mistralai==1.9.11
langchain-mistralai==0.2.12

# HuggingFace
huggingface-hub==1.1.2
langchain-huggingface==0.3.1

# GigaChat
langchain-gigachat==0.3.12
gigachat==0.1.42.post2

# Search Tools
langchain-tavily==0.2.13

# =============================================================================
# DATA PROCESSING & ANALYSIS
# =============================================================================

# Core Data Science
numpy==2.3.4
pandas==2.3.3
scikit-learn==1.7.2

# Image Processing
pillow==12.0.0

# Optional: Advanced Data Processing
matplotlib==3.10.7  # Optional - with fallback in code
pytesseract==0.3.13  # Optional - OCR functionality

# =============================================================================
# WEB & API INTEGRATIONS
# =============================================================================

# HTTP Requests
requests==2.32.5

# Configuration
python-dotenv==1.2.1
PyYAML==6.0.3

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

# PDF Processing - Best library for LLM processing
pymupdf4llm==0.1.7

# Excel Processing
openpyxl>=3.1.0

# Web Scraping (optional)
beautifulsoup4==4.14.2

# Markdown Processing
Markdown==3.10

# =============================================================================
# OBSERVABILITY & MONITORING
# =============================================================================

# LangSmith - Primary observability
langsmith==0.4.41

# Langfuse - Alternative observability
langfuse==3.9.0

# =============================================================================
# TOKEN COUNTING & UTILITIES
# =============================================================================

# Token Counting
tiktoken==0.12.0

# Data Validation
pydantic==2.12.4

# =============================================================================
# SEARCH & RESEARCH TOOLS
# =============================================================================

# Wikipedia
wikipedia==1.4.0

# ArXiv
arxiv==2.3.0

# =============================================================================
# DATABASE & STORAGE (Optional)
# =============================================================================

# Supabase (optional - for vector storage)
supabase>=2.0.0
pgvector>=0.2.0

# =============================================================================
# DEVELOPMENT & TESTING
# =============================================================================

# Testing Framework
pytest>=7.0.0
pytest-asyncio>=0.21.0

# Code Quality
ruff>=0.1.0  # Fast, modern linter (replaces black + flake8)
black>=23.0.0  # Code formatter
flake8>=6.0.0  # Linter

# =============================================================================
# OPTIONAL ADVANCED FEATURES
# =============================================================================

# MQTT Support (if needed)
asyncio-mqtt>=0.11.0

# Advanced ML (optional)
# sentence-transformers>=2.2.0  # Uncomment if vector similarity needed
# faiss-cpu>=1.7.4  # Uncomment if vector similarity needed
```

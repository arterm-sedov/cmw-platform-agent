# arterm-sedov Setup Instructions

## Overview

Welcome to the arterm-sedov CMW Platform Agent project! This guide ensures a smooth setup for both Windows and Linux/macOS, leveraging robust multi-LLM orchestration, model-level tool support, and transparent initialization diagnostics.

## Prerequisites

- **Python 3.8 or higher**
- **Git** (for cloning)
- **Internet connection**

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd arterm-sedov

# Run the automated setup script
python setup_venv.py
```

This script will:
- Detect your platform and Python version
- Create a virtual environment
- Use the correct requirements file for your OS
- Install all dependencies in order
- Verify installation and print next steps
- Print a summary of LLM/model initialization and tool support

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Dependencies

**For Windows:**
```bash
python -m pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.win.txt
```

**For Linux/macOS:**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Requirements Files

- `requirements.txt`: For Linux/macOS/Hugging Face Spaces
- `requirements.win.txt`: For Windows (handles platform quirks)

The setup script auto-selects the right file for you.

## Environment Variables Setup

Create a `.env` file in the project root:

```env
# Required for Google Gemini integration
GEMINI_KEY=your_gemini_api_key_here
# Required for Supabase vector store
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
# Optional: For HuggingFace, OpenRouter, Groq
HUGGINGFACEHUB_API_TOKEN=your_hf_token
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key
```

### Getting API Keys

- **Google Gemini:** [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Supabase:** [supabase.com](https://supabase.com) > Settings > API
- **HuggingFace:** [HuggingFace Tokens](https://huggingface.co/settings/tokens)

## Vector Store Setup

```bash
python setup_vector_store.py
```
This loads reference Q&A into Supabase for similarity search.

## Running the Agent

```bash
python app.py
```
This launches the Gradio web interface for interactive testing and evaluation.

## LLM Initialization & Tool Support

- On startup, each LLM/model is tested for plain and tool-calling support.
- **Google Gemini** is always bound with tools if enabled, even if the tool test returns empty (tool-calling works in practice; a warning is logged for transparency).
- **OpenRouter, Groq, and HuggingFace** are supported with model-level tool-calling detection and fallback.
- After initialization, a summary table is printed showing provider, model, plain/tools status, and any errors—so you always know what's available.

## Troubleshooting

### Common Issues

1. **Wrong requirements file used:**
   - The setup script auto-detects your platform. To force a file:
     ```bash
     pip install -r requirements.win.txt  # Windows
     pip install -r requirements.txt      # Linux/macOS
     ```
2. **Virtual environment creation fails:**
   - Remove and recreate:
     ```bash
     rm -rf venv  # Linux/macOS
     rmdir /s /q venv  # Windows
     python setup_venv.py
     ```
3. **Permission errors:**
   - Use `--user` flag:
     ```bash
     pip install --user -r requirements.txt
     ```
4. **Import errors after install:**
   - Reinstall dependencies:
     ```bash
     pip install --force-reinstall -r requirements.txt
     ```
5. **API key issues:**
   - Check your `.env` file for correct format and valid keys.

### Platform-Specific Issues

**Windows:**
- PowerShell execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Visual Studio Build Tools may be required for TensorFlow. Or use conda:
  ```cmd
  conda install pandas numpy
  pip install -r requirements.win.txt
  ```

**Linux/macOS:**
- Install system packages if needed:
  ```bash
  sudo apt-get install python3-dev build-essential  # Ubuntu/Debian
  xcode-select --install  # macOS
  ```

## Verification

After setup, verify everything works:

```python
import numpy as np
import pandas as pd
import langchain
import supabase
import gradio
print("✅ All core packages imported successfully!")
print(f"Pandas version: {pd.__version__}")
```

## Project Structure

```
arterm-sedov/
├── agent.py              # Main agent implementation
├── app.py                # Gradio web interface
├── tools.py              # Tool functions for the agent
├── setup_venv.py         # Cross-platform setup script
├── setup_vector_store.py # Vector store initialization
├── requirements.txt      # Dependencies (Linux/macOS/HF Space)
├── requirements.win.txt  # Dependencies (Windows)
├── system_prompt.txt     # Agent system prompt
├── metadata.jsonl        # Reference Q&A data
├── supabase_docs.csv     # Vector store backup
└── .env                  # Environment variables (create this)
```

## Advanced Configuration

### Custom Model Providers

The agent supports multiple LLM providers with robust fallback and model-level tool support:
- **Google Gemini**: Always bound with tools if enabled (tool-calling works even if test is empty)
- **Groq, OpenRouter, HuggingFace**: Model-level tool-calling detection and fallback

### Vector Store Configuration
- **Table:** `agent_course_reference`
- **Embedding Model:** `sentence-transformers/all-mpnet-base-v2`
- **Similarity Search:** Cosine similarity

### Tool Configuration
- Math, web, file, image, chess, code, and more—modular and extensible

## Support

- See the summary table after startup for LLM/model/tool status
- Review error logs for diagnostics
- For advanced help, see the troubleshooting section above

## Next Steps

1. **Test the agent** with sample questions
2. **Run the evaluation** for performance metrics
3. **Submit to CMW Platform Agent benchmark** for scoring
4. **Customize the agent** for your needs

The agent is now ready for the CMW Platform benchmark—battle-tested, transparent, and extensible. 🚀
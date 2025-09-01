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
# Optional: Supabase vector store (disabled by default)
# SUPABASE_URL=your_supabase_url_here
# SUPABASE_KEY=your_supabase_key_here
# Optional: Dataset uploading (disabled by default)
# HF_TOKEN=your_hf_token_here
# DATASET_ENABLED=true
# Optional: File uploading and HuggingFace login (disabled by default)
# FILE_UPLOAD_ENABLED=true
# LOGIN_ENABLED=true
# Optional: For HuggingFace, OpenRouter, Groq, Mistral AI
HUGGINGFACEHUB_API_TOKEN=your_hf_token
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_api_key
```

### Getting API Keys

- **Google Gemini:** [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Mistral AI:** [Mistral AI Platform](https://console.mistral.ai/) > API Keys
- **Supabase:** [supabase.com](https://supabase.com) > Settings > API
- **HuggingFace:** [HuggingFace Tokens](https://huggingface.co/settings/tokens)
- **OpenRouter:** [OpenRouter](https://openrouter.ai/) > API Keys
- **Groq:** [Groq Console](https://console.groq.com/) > API Keys

## Vector Store Setup (Optional)

The vector store functionality is disabled by default. To enable it:

1. **Install Supabase dependencies:**
   ```bash
   pip install supabase pgvector
   ```

2. **Set environment variables:**
   ```bash
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_KEY=your_supabase_key_here
   ```

3. **Enable in vector_store.py:**
   ```python
   SUPABASE_ENABLED = True
   ```

4. **Setup vector store:**
   ```bash
   python setup_vector_store.py
   ```
   This loads reference Q&A into Supabase for similarity search.

## Dataset Uploading Setup (Optional)

The dataset uploading functionality is disabled by default. To enable it:

1. **Install HuggingFace Hub dependency:**
   ```bash
   pip install huggingface_hub
   ```

2. **Set environment variables:**
   ```bash
   HF_TOKEN=your_hf_token_here
   DATASET_ENABLED=true
   ```

3. **Enable in dataset_manager.py:**
   ```python
   DATASET_ENABLED = True
   ```

The dataset manager will automatically upload:
- LLM initialization summaries to the "init" split
- Evaluation run data to the "runs_new" split

## File Uploading and Login Setup (Optional)

The file uploading and HuggingFace login functionality is disabled by default. To enable it:

1. **Install HuggingFace Hub dependency:**
   ```bash
   pip install huggingface_hub
   ```

2. **Set environment variables:**
   ```bash
   HF_TOKEN=your_hf_token_here
   FILE_UPLOAD_ENABLED=true
   LOGIN_ENABLED=true
   ```

3. **Enable in respective managers:**
   ```python
   # In file_manager.py
   FILE_UPLOAD_ENABLED = True
   
   # In login_manager.py
   LOGIN_ENABLED = True
   ```

The file manager will automatically upload:
- CSV results files to the HuggingFace Space repository
- Other generated files as needed

The login manager will:
- Enable HuggingFace login button in Gradio interface
- Validate user authentication for operations requiring login

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

# Optional: Test vector store functionality
try:
    from vector_store import get_status
    status = get_status()
    print(f"Vector store status: {status}")
except ImportError:
    print("ℹ️ Vector store module not available")

# Optional: Test dataset functionality
try:
    from dataset_manager import get_dataset_status
    status = get_dataset_status()
    print(f"Dataset manager status: {status}")
except ImportError:
    print("ℹ️ Dataset manager module not available")

# Optional: Test file manager functionality
try:
    from file_manager import get_file_manager_status
    status = get_file_manager_status()
    print(f"File manager status: {status}")
except ImportError:
    print("ℹ️ File manager module not available")

# Optional: Test login manager functionality
try:
    from login_manager import get_login_manager_status
    status = get_login_manager_status()
    print(f"Login manager status: {status}")
except ImportError:
    print("ℹ️ Login manager module not available")
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
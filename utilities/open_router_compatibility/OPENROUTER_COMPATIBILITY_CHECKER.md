# OpenRouter API Compatibility Checker

This tool checks the compatibility of our configured LLM models with the OpenRouter API by fetching model parameters and comparing them with our implementation.

## Features

- **Comprehensive Model Analysis**: Checks all models in our LLM configuration
- **Parameter Compatibility**: Compares supported parameters with our expected features
- **Detailed Reporting**: Generates markdown reports with compatibility scores and recommendations
- **Tool Support Detection**: Identifies which models support tool calling
- **Pricing Information**: Shows cost information for each model

## Usage

### Prerequisites

1. **OpenRouter API Key**: Get one from [OpenRouter](https://openrouter.ai/)
2. **Environment Setup**: Add your API key to a `.env` file in the project root

Create a `.env` file in the project root directory:
```bash
# .env file
OPENROUTER_API_KEY=your_api_key_here
```

Example `.env` file:
```bash
# OpenRouter API Configuration
# Get your API key from: https://openrouter.ai/
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Other API keys (if needed)
# GEMINI_KEY=your_gemini_key_here
# GROQ_API_KEY=your_groq_key_here
# MISTRAL_API_KEY=your_mistral_key_here
# HUGGINGFACE_API_KEY=your_huggingface_key_here
# GIGACHAT_API_KEY=your_gigachat_key_here
```

Alternatively, you can set the environment variable directly:
```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your_api_key_here"

# Windows Command Prompt
set OPENROUTER_API_KEY=your_api_key_here

# Linux/Mac
export OPENROUTER_API_KEY=your_api_key_here
```

### Running the Checker

#### Option 1: Direct Script Execution
```bash
cd utilities
python openrouter_compatibility_check.py
```

#### Option 2: Using the Runner Script
```bash
cd utilities
python run_openrouter_check.py
```

#### Option 3: From Project Root
```bash
python utilities/openrouter_compatibility_check.py
```

## What It Checks

### Model Availability
- Verifies if our configured models are available on OpenRouter
- Shows which models are missing from the OpenRouter catalog

### Parameter Compatibility
The checker compares our expected parameters with OpenRouter's supported parameters:

- `tools` - Function calling capabilities
- `tool_choice` - Tool selection control  
- `max_tokens` - Response length limiting
- `temperature` - Randomness control
- `top_p` - Nucleus sampling
- `stop` - Custom stop sequences
- `frequency_penalty` - Repetition reduction
- `presence_penalty` - Topic diversity
- `seed` - Deterministic outputs

### Context Length
- Compares our configured token limits with OpenRouter's context length
- Identifies potential context length mismatches

### Tool Support
- Checks if models support tool calling
- Identifies models that should have tool support disabled

## Output

### Console Output
The script displays:
- Number of models found in our configuration
- Number of models available on OpenRouter
- Compatibility scores for each model
- Summary statistics

### Generated Report
A detailed markdown report is saved to `docs/openrouter_compatibility_report_YYYYMMDD_HHMMSS.md` containing:

1. **Summary Statistics**
   - Total models checked
   - Available models count
   - Compatibility distribution

2. **Detailed Results**
   - Model-by-model analysis
   - Compatibility scores
   - Supported parameters
   - Missing/extra features
   - Pricing information

3. **Recommendations**
   - High compatibility models (recommended)
   - Models missing tool support
   - Unavailable models

## Example Output

```
OpenRouter API Compatibility Checker
====================================
Found 20 models in our configuration:
  - Google Gemini: gemini-2.5-pro
  - Groq: groq/compound
  - OpenRouter: openrouter/sonoma-dusk-alpha
  ...

Fetching OpenRouter models...
Found 400+ models on OpenRouter
Checking compatibility for 20 of our models...

# OpenRouter API Compatibility Report
Generated on: 2024-01-15 14:30:25

## Summary
- Total models checked: 20
- Available on OpenRouter: 15
- High compatibility (≥80%): 12
- Medium compatibility (50-79%): 3
- Low compatibility (<50%): 5

## Detailed Results
...
```

## Configuration

The checker automatically extracts models from our `LLMManager` configuration in `agent_ng/llm_manager.py`. It checks:

- **OpenRouter models**: All models configured for OpenRouter provider
- **Other providers**: Models from Gemini, Groq, Mistral, etc. (to see if they're available on OpenRouter)

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: OPENROUTER_API_KEY environment variable not set
   ```
   Solution: Set the environment variable as shown in the Prerequisites section.

2. **Import Errors**
   ```
   ImportError: Could not import LLMManager
   ```
   Solution: Run from the correct directory or ensure the project structure is intact.

3. **Network Issues**
   ```
   Error fetching OpenRouter models: Connection timeout
   ```
   Solution: Check your internet connection and try again.

### Testing

Run the test suite to verify everything works:

```bash
cd utilities
python test_openrouter_check.py
```

## Integration

This tool is designed to be run periodically to:
- Verify model compatibility after OpenRouter updates
- Check new models before adding them to our configuration
- Validate our implementation against OpenRouter's current capabilities
- Generate reports for documentation and decision-making

## Files

- `utilities/openrouter_compatibility_check.py` - Main checker script
- `utilities/run_openrouter_check.py` - Simple runner with error handling
- `utilities/test_openrouter_check.py` - Test suite
- `docs/openrouter_compatibility_report_*.md` - Generated reports

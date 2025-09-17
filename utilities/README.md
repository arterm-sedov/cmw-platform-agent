# Utilities

This directory contains utility scripts and tools for the CMW Platform Agent project.

## OpenRouter Compatibility Checker

A comprehensive tool to check the compatibility of our configured LLM models with the OpenRouter API.

### Location
`utilities/open_router_compatibility/`

### Files

- `openrouter_compatibility_check.py` - Main checker script
- `run_openrouter_check.py` - Simple runner with error handling  
- `test_openrouter_check.py` - Test suite
- `OPENROUTER_COMPATIBILITY_CHECKER.md` - Detailed documentation
- `openrouter_compatibility_report_*.md` - Generated compatibility reports

### Usage

```bash
# From open_router_compatibility directory
cd utilities/open_router_compatibility
python openrouter_compatibility_check.py

# Or use the runner
python run_openrouter_check.py

# Run tests
python test_openrouter_check.py
```

### Features

- **Model Compatibility Analysis**: Checks all 20+ models in our configuration
- **Parameter Support Detection**: Verifies tool calling, temperature, max_tokens, etc.
- **Detailed Reporting**: Generates markdown reports with recommendations
- **OpenRouter API Integration**: Fetches real-time model data from OpenRouter

### Prerequisites

1. Set up your OpenRouter API key in a `.env` file:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```

2. Install required dependencies:
   ```bash
   pip install requests python-dotenv
   ```

### Generated Reports

Reports are automatically saved to `../docs/openrouter_compatibility_report_*.md` with:
- Compatibility scores for each model
- Missing/extra features analysis
- Tool support verification
- Pricing information
- Actionable recommendations

## Other Utilities

Additional utility scripts will be added to this directory as needed for:
- Model testing and validation
- Configuration management
- Performance monitoring
- Debugging tools

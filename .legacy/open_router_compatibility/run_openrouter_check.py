#!/usr/bin/env python3
"""
Simple runner script for OpenRouter compatibility check
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Run the OpenRouter compatibility check"""
    print("OpenRouter API Compatibility Checker")
    print("=" * 40)
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nTo use this script:")
        print("1. Get an API key from https://openrouter.ai/")
        print("2. Add it to a .env file in the project root:")
        print("   OPENROUTER_API_KEY=your_key_here")
        print("3. Or set the environment variable directly:")
        print("   - Windows: set OPENROUTER_API_KEY=your_key_here")
        print("   - Linux/Mac: export OPENROUTER_API_KEY=your_key_here")
        print("4. Run this script again")
        return 1
    
    # Import and run the checker
    try:
        from openrouter_compatibility_check import main as run_checker
        run_checker()
        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the .misc_files directory")
        return 1
    except Exception as e:
        print(f"❌ Error running checker: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

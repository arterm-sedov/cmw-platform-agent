#!/usr/bin/env python3
"""
Comprehensive analysis to verify all error codes from agent.py are handled in error_handler.py
"""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_error_codes_from_agent():
    """Extract all error codes mentioned in agent.py"""
    agent_codes = set()
    
    # Read agent.py and find all status codes
    with open('agent.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all HTTP status codes in the file
    patterns = [
        r'status_code\s*==\s*(\d{3})',  # status_code == 400
        r'status.*?(\d{3})',            # status: 400
        r'code.*?(\d{3})',              # code: 400
        r'HTTP\s+(\d{3})',              # HTTP 400
        r'(\d{3})\s+error',             # 400 error
        r'for status in \[([^\]]+)\]',  # for status in [400, 401, ...]
        r'"(\d{3})"',                   # "400"
        r'(\d{3})',                     # Any 3-digit number
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if isinstance(match, str) and ',' in match:
                # Handle lists like "400, 401, 402, 403, 404, 408, 413, 422, 429, 500, 502, 503"
                codes = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
                agent_codes.update(codes)
            elif match.isdigit():
                agent_codes.add(int(match))
    
    # Filter to valid HTTP status codes (400-599)
    agent_codes = {code for code in agent_codes if 400 <= code <= 599}
    
    return sorted(agent_codes)

def extract_error_codes_from_error_handler():
    """Extract all error codes handled in error_handler.py"""
    handler_codes = set()
    
    # Read error_handler.py and find all status codes
    with open('error_handler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all status code checks
    patterns = [
        r'status_code\s*==\s*(\d{3})',  # status_code == 400
        r'for status in \[([^\]]+)\]',  # for status in [400, 401, ...]
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if isinstance(match, str) and ',' in match:
                # Handle lists
                codes = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
                handler_codes.update(codes)
            elif match.isdigit():
                handler_codes.add(int(match))
    
    # Filter to valid HTTP status codes (400-599)
    handler_codes = {code for code in handler_codes if 400 <= code <= 599}
    
    return sorted(handler_codes)

def analyze_coverage():
    """Analyze error code coverage between agent.py and error_handler.py"""
    print("🔍 Analyzing Error Code Coverage\n")
    
    agent_codes = extract_error_codes_from_agent()
    handler_codes = extract_error_codes_from_error_handler()
    
    print(f"📊 Error codes found in agent.py: {len(agent_codes)}")
    print(f"   {agent_codes}")
    
    print(f"\n📊 Error codes handled in error_handler.py: {len(handler_codes)}")
    print(f"   {handler_codes}")
    
    # Find missing codes
    missing_in_handler = set(agent_codes) - set(handler_codes)
    missing_in_agent = set(handler_codes) - set(agent_codes)
    
    print(f"\n✅ Coverage Analysis:")
    print(f"   Total codes in agent.py: {len(agent_codes)}")
    print(f"   Total codes in error_handler.py: {len(handler_codes)}")
    print(f"   Codes covered: {len(set(agent_codes) & set(handler_codes))}")
    print(f"   Coverage percentage: {len(set(agent_codes) & set(handler_codes)) / len(agent_codes) * 100:.1f}%")
    
    if missing_in_handler:
        print(f"\n❌ Missing in error_handler.py: {sorted(missing_in_handler)}")
    else:
        print(f"\n✅ All error codes from agent.py are covered in error_handler.py!")
    
    if missing_in_agent:
        print(f"\n➕ Additional codes in error_handler.py: {sorted(missing_in_agent)}")
    
    # Check for specific important codes
    important_codes = [400, 401, 402, 403, 404, 413, 422, 429, 498, 499, 500, 502, 503, 504]
    missing_important = [code for code in important_codes if code not in handler_codes]
    
    if missing_important:
        print(f"\n⚠️ Missing important HTTP codes: {missing_important}")
    else:
        print(f"\n✅ All important HTTP error codes are covered!")
    
    return len(missing_in_handler) == 0

def check_provider_specific_coverage():
    """Check if all provider-specific error handling is covered"""
    print(f"\n🔍 Checking Provider-Specific Error Coverage\n")
    
    providers = ['gemini', 'groq', 'mistral', 'gigachat', 'openrouter', 'huggingface']
    
    with open('error_handler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    for provider in providers:
        if f'classify_{provider}_error' in content:
            print(f"✅ {provider.upper()}: Has specific error classification")
        else:
            print(f"❌ {provider.upper()}: Missing specific error classification")

def main():
    """Run the comprehensive analysis"""
    print("🚀 Error Code Coverage Analysis\n")
    
    try:
        # Check basic coverage
        is_fully_covered = analyze_coverage()
        
        # Check provider-specific coverage
        check_provider_specific_coverage()
        
        print(f"\n📋 Summary:")
        if is_fully_covered:
            print("✅ All error codes from agent.py are now handled in error_handler.py!")
            print("✅ The error handler provides comprehensive coverage.")
        else:
            print("❌ Some error codes from agent.py are missing from error_handler.py")
            print("❌ Additional work needed to achieve full coverage.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for Hugging Face Spaces API endpoints
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

def test_hf_api_endpoints():
    """Test API endpoints on Hugging Face Spaces"""

    # Get HF Space URL from environment or use default
    hf_space_url = os.getenv("HF_SPACE_URL", "https://arterm-sedov-cmw-agent.hf.space")

    print(f"Testing Hugging Face Space: {hf_space_url}")

    # Test 1: Check if the space is accessible
    try:
        response = requests.get(hf_space_url, timeout=10)
        print(f"✅ Space accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Space not accessible: {e}")
        return

    # Test 2: Check API documentation
    try:
        api_url = f"{hf_space_url}/?view=api"
        response = requests.get(api_url, timeout=10)
        print(f"✅ API docs accessible: {response.status_code}")
        if "No API Routes found" in response.text:
            print("⚠️  No API routes found - this might be the issue")
        else:
            print("✅ API routes found")
    except Exception as e:
        print(f"❌ API docs not accessible: {e}")

    # Test 3: Test /ask endpoint with different URL formats
    test_urls = [
        f"{hf_space_url}/ask",  # New gr.api() format
        f"{hf_space_url}/call/ask",  # Original format
        f"{hf_space_url}/api/predict",  # Alternative format
    ]

    for i, ask_url in enumerate(test_urls):
        try:
            # Try GET request with query parameters (gr.api() uses GET by default)
            params = {
                "question": "Hello, test API from HF Space",
                "session_hash": "test-session-123"
            }
            response = requests.get(ask_url, params=params, timeout=30)
            print(f"✅ /ask endpoint GET (format {i+1}): {response.status_code}")
            if response.status_code == 200:
                print(f"   Response text: {response.text[:200]}...")
                try:
                    result = response.json()
                    print(f"   Response JSON: {result}")
                    break
                except Exception as json_error:
                    print(f"   JSON Error: {json_error}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ /ask endpoint GET error (format {i+1}): {e}")

        try:
            # Try POST request as fallback
            payload = {
                "data": ["Hello, test API from HF Space"],
                "session_hash": "test-session-123"
            }
            response = requests.post(ask_url, json=payload, timeout=30)
            print(f"✅ /ask endpoint POST (format {i+1}): {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Response: {result}")
                break
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ /ask endpoint POST error (format {i+1}): {e}")

    # Test 4: Test /ask_stream endpoint with different URL formats
    stream_urls = [
        f"{hf_space_url}/ask_stream",  # New gr.api() format
        f"{hf_space_url}/call/ask_stream",  # Original format
        f"{hf_space_url}/api/predict",  # Alternative format
    ]

    for i, stream_url in enumerate(stream_urls):
        try:
            # Try GET request with query parameters (gr.api() uses GET by default)
            params = {
                "question": "Hello streaming from HF Space",
                "session_hash": "test-session-456"
            }
            response = requests.get(stream_url, params=params, timeout=30)
            print(f"✅ /ask_stream endpoint GET (format {i+1}): {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Response: {result}")
                break
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ /ask_stream endpoint GET error (format {i+1}): {e}")

        try:
            # Try POST request as fallback
            payload = {
                "data": ["Hello streaming from HF Space"],
                "session_hash": "test-session-456"
            }
            response = requests.post(stream_url, json=payload, timeout=30)
            print(f"✅ /ask_stream endpoint POST (format {i+1}): {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Response: {result}")
                break
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ /ask_stream endpoint POST error (format {i+1}): {e}")

if __name__ == "__main__":
    test_hf_api_endpoints()

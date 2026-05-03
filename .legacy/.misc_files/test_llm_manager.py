"""
Test script for LLM Manager Module
==================================

This script tests the LLM Manager functionality to ensure it works correctly.
"""

import os
import sys
from agent_ng.llm_manager import get_llm_manager, LLMProvider, reset_llm_manager


def test_llm_manager():
    """Test the LLM Manager functionality"""
    print("Testing LLM Manager...")

    # Reset manager for clean test
    reset_llm_manager()
    manager = get_llm_manager()

    # Test 1: Get available providers
    print("\n1. Testing available providers...")
    available = manager.get_available_providers()
    print(f"Available providers: {available}")

    # Test 2: Get provider config
    print("\n2. Testing provider configuration...")
    for provider in ["gemini", "groq", "huggingface"]:
        config = manager.get_provider_config(provider)
        if config:
            print(f"{provider}: {config.name} - {len(config.models)} models")
        else:
            print(f"{provider}: Not available")

    # Test 3: Try to get an LLM instance
    print("\n3. Testing LLM instance retrieval...")
    test_providers = ["gemini", "groq", "openrouter"]

    for provider in test_providers:
        print(f"\nTrying to get {provider} LLM...")
        instance = manager.get_llm(provider)
        if instance:
            print(f"✓ Successfully got {provider} - {instance.model_name}")
            print(f"  Provider: {instance.provider.value}")
            print(f"  Healthy: {instance.is_healthy}")
            print(f"  Config: {instance.config}")
        else:
            print(f"✗ Failed to get {provider} LLM")

    # Test 4: Get LLM sequence
    print("\n4. Testing LLM sequence...")
    sequence = manager.get_llm_sequence()
    print(f"LLM sequence: {[inst.provider.value for inst in sequence]}")

    # Test 5: Health check
    print("\n5. Testing health check...")
    health = manager.health_check()
    print(f"Health check result: {health}")

    # Test 6: Get stats
    print("\n6. Testing stats...")
    stats = manager.get_stats()
    print(f"Stats: {stats}")

    # Test 7: Get initialization logs
    print("\n7. Testing initialization logs...")
    logs = manager.get_initialization_logs()
    print(f"Initialization logs ({len(logs)} entries):")
    for log in logs[-5:]:  # Show last 5 logs
        print(f"  {log}")

    print("\n✓ LLM Manager test completed!")


if __name__ == "__main__":
    test_llm_manager()

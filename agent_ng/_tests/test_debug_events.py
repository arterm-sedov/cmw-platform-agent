#!/usr/bin/env python3
"""
Test script to demonstrate event-based vs timer-based debug logging.

This script shows how debug messages are now only shown on actual events
rather than being repeated by timers.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_ng.langchain_agent import CmwAgent
from agent_ng.langchain_memory import ConversationMemoryManager

def test_debug_events():
    """Test that debug messages only appear on events, not on timer ticks"""
    
    print("🧪 Testing Event-Based Debug Logging")
    print("=" * 50)
    
    # Create agent and memory manager
    memory_manager = ConversationMemoryManager()
    agent = CmwAgent(memory_manager=memory_manager)
    
    print("\n1. Testing conversation stats without debug (timer simulation):")
    print("-" * 60)
    
    # Simulate timer-based calls (like the UI timer)
    for i in range(3):
        print(f"\nTimer tick {i+1}:")
        stats = agent._get_conversation_stats(debug=False)  # No debug output
        print(f"Stats: {stats}")
        time.sleep(0.1)
    
    print("\n2. Testing conversation stats with debug (event simulation):")
    print("-" * 60)
    
    # Simulate event-based calls (like when messages are added)
    print("\nEvent: New message added")
    stats = agent.get_conversation_stats(debug=True)  # Debug output enabled
    print(f"Stats: {stats}")
    
    print("\n3. Testing change detection in stats tab:")
    print("-" * 60)
    
    # Simulate stats tab behavior
    from agent_ng.tabs.stats_tab import StatsTab
    
    # Create a mock stats tab
    class MockStatsTab(StatsTab):
        def __init__(self):
            self.agent = agent
            self._last_conversation_stats = None
        
        def test_change_detection(self):
            """Test that debug only shows on changes"""
            print("\nFirst call (no previous stats):")
            conversation_stats = self.agent._get_conversation_stats(debug=False)
            debug_stats = self._last_conversation_stats != conversation_stats
            if debug_stats:
                print("🔍 Change detected - showing debug:")
                conversation_stats = self.agent._get_conversation_stats(debug=True)
            self._last_conversation_stats = conversation_stats.copy() if conversation_stats else None
            
            print("\nSecond call (same stats):")
            conversation_stats = self.agent._get_conversation_stats(debug=False)
            debug_stats = self._last_conversation_stats != conversation_stats
            if debug_stats:
                print("🔍 Change detected - showing debug:")
                conversation_stats = self.agent._get_conversation_stats(debug=True)
            else:
                print("No change detected - no debug output")
            self._last_conversation_stats = conversation_stats.copy() if conversation_stats else None
    
    mock_tab = MockStatsTab()
    mock_tab.test_change_detection()
    
    print("\n✅ Test completed!")
    print("\nSummary:")
    print("- Timer-based calls (every 3 seconds) now show no debug output")
    print("- Event-based calls (when messages change) show debug output")
    print("- Change detection prevents duplicate debug messages")

if __name__ == "__main__":
    test_debug_events()

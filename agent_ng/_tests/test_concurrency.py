"""
Concurrency Testing Module
=========================

A lean testing module for validating concurrent processing in the CMW Platform Agent.
This module provides utilities for testing the concurrency configuration and
queue management functionality.

Key Features:
- Concurrent request simulation
- Performance metrics collection
- Configuration validation
- Gradio-native testing patterns
"""

import asyncio
import time
import threading
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import gradio as gr
from dataclasses import dataclass
from datetime import datetime

from .concurrency_config import ConcurrencyConfig, get_concurrency_config
from .queue_manager import QueueManager, create_queue_manager


@dataclass
class ConcurrencyTestResult:
    """Result of a concurrency test"""
    test_name: str
    success: bool
    duration: float
    concurrent_requests: int
    successful_requests: int
    failed_requests: int
    error_message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ConcurrencyTester:
    """
    Lean concurrency tester for Gradio applications.
    
    This class provides utilities for testing concurrent processing
    capabilities of the CMW Platform Agent.
    """
    
    def __init__(self, config: ConcurrencyConfig = None):
        """
        Initialize the concurrency tester.
        
        Args:
            config: Concurrency configuration to test
        """
        self.config = config or get_concurrency_config()
        self.queue_manager = create_queue_manager(self.config)
        self.test_results: List[ConcurrencyTestResult] = []
    
    def test_configuration_validation(self) -> ConcurrencyTestResult:
        """
        Test that the concurrency configuration is valid.
        
        Returns:
            ConcurrencyTestResult with test results
        """
        start_time = time.time()
        success = True
        error_message = ""
        
        try:
            # Test configuration validation
            assert self.config.queue.default_concurrency_limit > 0
            assert self.config.queue.max_threads > 0
            assert self.config.events.chat_concurrency_limit > 0
            
            # Test queue manager creation
            assert self.queue_manager is not None
            assert self.queue_manager.config == self.config
            
        except Exception as e:
            success = False
            error_message = str(e)
        
        duration = time.time() - start_time
        
        result = ConcurrencyTestResult(
            test_name="Configuration Validation",
            success=success,
            duration=duration,
            concurrent_requests=0,
            successful_requests=1 if success else 0,
            failed_requests=0 if success else 1,
            error_message=error_message
        )
        
        self.test_results.append(result)
        return result
    
    def test_concurrent_requests(self, num_requests: int = 5, request_delay: float = 0.1) -> ConcurrencyTestResult:
        """
        Test concurrent request processing.
        
        Args:
            num_requests: Number of concurrent requests to simulate
            request_delay: Delay between request submissions
            
        Returns:
            ConcurrencyTestResult with test results
        """
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        error_message = ""
        
        def mock_request_handler(request_id: int) -> Dict[str, Any]:
            """Mock request handler for testing"""
            try:
                # Simulate some processing time
                time.sleep(0.1)
                return {"request_id": request_id, "status": "success"}
            except Exception as e:
                return {"request_id": request_id, "status": "error", "error": str(e)}
        
        try:
            # Submit concurrent requests
            with ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = []
                
                for i in range(num_requests):
                    future = executor.submit(mock_request_handler, i)
                    futures.append(future)
                    time.sleep(request_delay)  # Stagger requests slightly
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=5.0)
                        if result["status"] == "success":
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    except Exception as e:
                        failed_requests += 1
                        error_message += f"Request failed: {str(e)}; "
        
        except Exception as e:
            error_message = str(e)
            failed_requests = num_requests
        
        duration = time.time() - start_time
        success = failed_requests == 0
        
        result = ConcurrencyTestResult(
            test_name=f"Concurrent Requests ({num_requests})",
            success=success,
            duration=duration,
            concurrent_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_message=error_message
        )
        
        self.test_results.append(result)
        return result
    
    def test_gradio_queue_configuration(self) -> ConcurrencyTestResult:
        """
        Test that Gradio queue configuration is properly applied.
        
        Returns:
            ConcurrencyTestResult with test results
        """
        start_time = time.time()
        success = True
        error_message = ""
        
        try:
            # Create a mock Gradio interface
            with gr.Blocks() as demo:
                text_input = gr.Textbox(label="Test Input")
                text_output = gr.Textbox(label="Test Output")
                
                def test_function(text):
                    return f"Processed: {text}"
                
                text_input.submit(test_function, text_input, text_output)
            
            # Configure queue
            self.queue_manager.configure_queue(demo)
            
            # Verify configuration was applied
            queue_status = self.queue_manager.get_queue_status()
            assert queue_status['queue_configured'] == True
            assert queue_status['concurrent_processing_enabled'] == self.config.enable_concurrent_processing
            
        except Exception as e:
            success = False
            error_message = str(e)
        
        duration = time.time() - start_time
        
        result = ConcurrencyTestResult(
            test_name="Gradio Queue Configuration",
            success=success,
            duration=duration,
            concurrent_requests=0,
            successful_requests=1 if success else 0,
            failed_requests=0 if success else 1,
            error_message=error_message
        )
        
        self.test_results.append(result)
        return result
    
    def test_event_concurrency_settings(self) -> ConcurrencyTestResult:
        """
        Test that event-specific concurrency settings are properly configured.
        
        Returns:
            ConcurrencyTestResult with test results
        """
        start_time = time.time()
        success = True
        error_message = ""
        successful_requests = 0
        failed_requests = 0
        
        try:
            # Test different event types
            event_types = ['chat', 'file_upload', 'stats_refresh', 'logs_refresh']
            
            for event_type in event_types:
                concurrency_config = self.queue_manager.get_event_concurrency(event_type)
                
                # Verify configuration exists and is valid
                assert 'concurrency_limit' in concurrency_config
                assert concurrency_config['concurrency_limit'] > 0
                successful_requests += 1
            
        except Exception as e:
            success = False
            error_message = str(e)
            failed_requests = len(event_types)
        
        duration = time.time() - start_time
        
        result = ConcurrencyTestResult(
            test_name="Event Concurrency Settings",
            success=success,
            duration=duration,
            concurrent_requests=0,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_message=error_message
        )
        
        self.test_results.append(result)
        return result
    
    def run_all_tests(self) -> List[ConcurrencyTestResult]:
        """
        Run all concurrency tests.
        
        Returns:
            List of ConcurrencyTestResult objects
        """
        print("🧪 Starting concurrency tests...")
        
        # Run all tests
        self.test_configuration_validation()
        self.test_gradio_queue_configuration()
        self.test_event_concurrency_settings()
        self.test_concurrent_requests(num_requests=3)
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self) -> None:
        """Print a summary of all test results"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - successful_tests
        
        print(f"\n📊 Concurrency Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"   - {result.test_name}: {result.error_message}")
        
        print(f"\n⚡ Performance Metrics:")
        for result in self.test_results:
            if result.concurrent_requests > 0:
                print(f"   - {result.test_name}: {result.duration:.3f}s for {result.concurrent_requests} requests")


def run_concurrency_tests(config: ConcurrencyConfig = None) -> List[ConcurrencyTestResult]:
    """
    Convenience function to run all concurrency tests.
    
    Args:
        config: Optional concurrency configuration to test
        
    Returns:
        List of ConcurrencyTestResult objects
    """
    tester = ConcurrencyTester(config)
    return tester.run_all_tests()


if __name__ == "__main__":
    # Run tests when executed directly
    results = run_concurrency_tests()
    
    # Exit with appropriate code
    failed_tests = sum(1 for result in results if not result.success)
    exit(failed_tests)

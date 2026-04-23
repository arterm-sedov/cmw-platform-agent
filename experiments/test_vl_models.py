"""
Vision-Language Model Testing Script
=====================================

Test VL capabilities of models via OpenRouter API.

Usage:
    python experiments/test_vl_models.py

Requirements:
    - OPENROUTER_API_KEY environment variable set
    - Test files in experiments/test_files/ directory
"""

import os
import sys
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class VLModelTester:
    """Test vision-language models via OpenRouter"""

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.results = []

    def encode_file_to_base64(self, file_path: str) -> str:
        """Encode file to base64 string"""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type from file extension"""
        ext = file_path.split('.')[-1].lower()
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'mp4': 'video/mp4',
            'avi': 'video/x-msvideo',
            'mov': 'video/quicktime',
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'pdf': 'application/pdf'
        }
        return mime_types.get(ext, 'application/octet-stream')

    def test_model(
        self,
        model: str,
        file_path: str,
        prompt: str,
        file_type: str = "image"
    ) -> Dict[str, Any]:
        """
        Test a model with a file input.

        Args:
            model: Model identifier (e.g., "qwen/qwen3.6-plus")
            file_path: Path to test file
            prompt: Analysis prompt
            file_type: Type of file (image, video, audio, pdf)

        Returns:
            Test result dictionary
        """
        print(f"\n{'='*80}")
        print(f"Testing: {model}")
        print(f"File: {file_path}")
        print(f"Type: {file_type}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")

        result = {
            "model": model,
            "file_path": file_path,
            "file_type": file_type,
            "prompt": prompt,
            "timestamp": time.time(),
            "success": False,
            "response": None,
            "error": None,
            "latency_ms": None,
            "cost_estimate": None
        }

        try:
            # Encode file to base64
            print("Encoding file to base64...")
            base64_data = self.encode_file_to_base64(file_path)
            mime_type = self.detect_mime_type(file_path)

            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/arterm-sedov/cmw-platform-agent",
                "X-Title": "CMW Platform Agent VL Testing"
            }

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_data}"
                                }
                            }
                        ]
                    }
                ]
            }

            # Make request
            print("Sending request to OpenRouter...")
            start_time = time.time()
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            latency = (time.time() - start_time) * 1000  # Convert to ms

            result["latency_ms"] = latency

            # Check response
            if response.status_code == 200:
                data = response.json()
                result["success"] = True
                result["response"] = data["choices"][0]["message"]["content"]

                # Extract usage info if available
                if "usage" in data:
                    result["usage"] = data["usage"]

                print(f"\n✅ SUCCESS (latency: {latency:.0f}ms)")
                print(f"\nResponse:\n{result['response'][:500]}...")

            else:
                result["error"] = f"HTTP {response.status_code}: {response.text}"
                print(f"\n❌ FAILED: {result['error']}")

        except Exception as e:
            result["error"] = str(e)
            print(f"\n❌ ERROR: {e}")

        self.results.append(result)
        return result

    def save_results(self, output_file: str = "experiments/vl_test_results.json"):
        """Save test results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n\n📊 Results saved to: {output_file}")

    def print_summary(self):
        """Print summary of test results"""
        print(f"\n\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")

        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        print(f"\nTotal tests: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")

        if successful > 0:
            avg_latency = sum(r["latency_ms"] for r in self.results if r["success"]) / successful
            print(f"Average latency: {avg_latency:.0f}ms")

        print(f"\n{'='*80}")
        print("RESULTS BY MODEL")
        print(f"{'='*80}")

        models = {}
        for r in self.results:
            model = r["model"]
            if model not in models:
                models[model] = {"success": 0, "failed": 0, "latencies": []}

            if r["success"]:
                models[model]["success"] += 1
                models[model]["latencies"].append(r["latency_ms"])
            else:
                models[model]["failed"] += 1

        for model, stats in models.items():
            total_tests = stats["success"] + stats["failed"]
            success_rate = stats["success"] / total_tests * 100
            avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0

            print(f"\n{model}:")
            print(f"  Success rate: {success_rate:.1f}% ({stats['success']}/{total_tests})")
            if avg_latency > 0:
                print(f"  Avg latency: {avg_latency:.0f}ms")


def main():
    """Run VL model tests"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Vision-Language Model Testing Script                      ║
║                                                                              ║
║  This script tests VL capabilities of models via OpenRouter API             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize tester
    tester = VLModelTester()

    # Define test cases
    test_cases = [
        # Qwen 3.6 Plus tests
        {
            "model": "qwen/qwen3.6-plus",
            "file_path": "experiments/test_files/test_image.jpg",
            "prompt": "Describe this image in detail.",
            "file_type": "image"
        },
        {
            "model": "qwen/qwen3.6-plus",
            "file_path": "experiments/test_files/test_document.jpg",
            "prompt": "Extract all text from this image (OCR). Return only the extracted text.",
            "file_type": "image"
        },
        {
            "model": "qwen/qwen3.6-plus",
            "file_path": "experiments/test_files/test_chart.jpg",
            "prompt": "Analyze this chart and extract the data in a structured format.",
            "file_type": "image"
        },

        # Gemini 3.1 Flash tests (if available)
        {
            "model": "google/gemini-3.1-flash",
            "file_path": "experiments/test_files/test_image.jpg",
            "prompt": "Describe this image in detail.",
            "file_type": "image"
        },
        {
            "model": "google/gemini-3.1-flash",
            "file_path": "experiments/test_files/test_document.jpg",
            "prompt": "Extract all text from this image (OCR). Return only the extracted text.",
            "file_type": "image"
        },

        # Gemini 2.0 Flash tests (fallback)
        {
            "model": "google/gemini-2.0-flash-exp",
            "file_path": "experiments/test_files/test_image.jpg",
            "prompt": "Describe this image in detail.",
            "file_type": "image"
        },
        {
            "model": "google/gemini-2.0-flash-exp",
            "file_path": "experiments/test_files/test_document.jpg",
            "prompt": "Extract all text from this image (OCR). Return only the extracted text.",
            "file_type": "image"
        },

        # Xiaomi Mimo v2.5 tests
        {
            "model": "xiaomi/mimo-v2.5",
            "file_path": "experiments/test_files/test_image.jpg",
            "prompt": "Describe this image in detail.",
            "file_type": "image"
        },
    ]

    # Check if test files exist
    test_files_dir = Path("experiments/test_files")
    if not test_files_dir.exists():
        print(f"\n⚠️  Test files directory not found: {test_files_dir}")
        print("\nPlease create the directory and add test files:")
        print("  - test_image.jpg (any image)")
        print("  - test_document.jpg (image with text for OCR)")
        print("  - test_chart.jpg (image with chart/graph)")
        print("  - test_video.mp4 (optional, short video)")
        print("  - test_audio.mp3 (optional, audio file)")
        return

    # Run tests
    print(f"\n🚀 Starting tests...")
    print(f"Total test cases: {len(test_cases)}")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n[Test {i}/{len(test_cases)}]")

        # Check if file exists
        if not Path(test_case["file_path"]).exists():
            print(f"⚠️  Skipping: File not found - {test_case['file_path']}")
            continue

        # Run test
        tester.test_model(**test_case)

        # Small delay between requests
        if i < len(test_cases):
            time.sleep(2)

    # Print summary
    tester.print_summary()

    # Save results
    tester.save_results()

    print("\n\n✅ Testing complete!")
    print("\nNext steps:")
    print("1. Review results in experiments/vl_test_results.json")
    print("2. Compare OCR quality with Tesseract")
    print("3. Update VL_MODEL_SUPPORT_IMPLEMENTATION.md with findings")
    print("4. Proceed with Phase 1 implementation")


if __name__ == "__main__":
    main()

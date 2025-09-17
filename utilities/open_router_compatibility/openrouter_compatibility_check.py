#!/usr/bin/env python3
"""
OpenRouter API Compatibility Checker
====================================

This script checks the compatibility of our configured models with OpenRouter's API
by fetching model parameters and comparing them with our implementation.

Usage:
    python openrouter_compatibility_check.py

Requirements:
    - OPENROUTER_API_KEY environment variable
    - requests library
"""

import os
import json
import requests
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(utilities_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from agent_ng.llm_manager import LLMManager, LLMProvider, LLMConfig
except ImportError:
    print("Error: Could not import LLMManager. Make sure you're running from the correct directory.")
    sys.exit(1)


@dataclass
class ModelCompatibility:
    """Represents compatibility information for a model"""
    model_id: str
    name: str
    context_length: int
    supported_parameters: List[str]
    pricing: Dict[str, str]
    architecture: Dict[str, Any]
    is_available: bool
    compatibility_score: float
    missing_features: List[str]
    extra_features: List[str]


class OpenRouterCompatibilityChecker:
    """Checks compatibility between our models and OpenRouter API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("Error: OPENROUTER_API_KEY environment variable not set")
            sys.exit(1)
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Our expected parameters that we use
        self.our_expected_params = [
            "tools", "tool_choice", "max_tokens", "temperature", 
            "top_p", "stop", "frequency_penalty", "presence_penalty", "seed"
        ]
        
        # Initialize our LLM manager to get our models
        self.llm_manager = LLMManager()
        self.our_models = self._extract_our_models()
    
    def _extract_our_models(self) -> List[Dict[str, Any]]:
        """Extract all models from our LLM configuration"""
        models = []
        
        for provider, config in self.llm_manager.LLM_CONFIGS.items():
            for model_config in config.models:
                models.append({
                    "provider": provider.value,
                    "provider_name": config.name,
                    "model_id": model_config["model"],
                    "token_limit": model_config.get("token_limit", 0),
                    "max_tokens": model_config.get("max_tokens", 0),
                    "temperature": model_config.get("temperature", 0),
                    "tool_support": config.tool_support,
                    "force_tools": config.force_tools
                })
        
        return models
    
    def fetch_openrouter_models(self) -> Dict[str, Any]:
        """Fetch all available models from OpenRouter API"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching OpenRouter models: {e}")
            return {"data": []}
    
    def find_model_in_openrouter(self, model_id: str, openrouter_models: List[Dict]) -> Optional[Dict]:
        """Find a specific model in OpenRouter's model list"""
        for model in openrouter_models:
            if model.get("id") == model_id:
                return model
        return None
    
    def calculate_compatibility_score(self, our_model: Dict, openrouter_model: Optional[Dict]) -> float:
        """Calculate compatibility score between our model and OpenRouter model"""
        if not openrouter_model:
            return 0.0
        
        score = 0.0
        total_checks = 0
        
        # Check context length compatibility
        if "context_length" in openrouter_model:
            our_limit = our_model.get("token_limit", 0)
            openrouter_limit = openrouter_model["context_length"]
            if our_limit <= openrouter_limit:
                score += 1.0
            total_checks += 1
        
        # Check supported parameters
        supported_params = openrouter_model.get("supported_parameters", [])
        param_matches = 0
        for param in self.our_expected_params:
            if param in supported_params:
                param_matches += 1
        if self.our_expected_params:
            score += (param_matches / len(self.our_expected_params))
        total_checks += 1
        
        # Check tool support
        if our_model.get("tool_support", False):
            if "tools" in supported_params:
                score += 1.0
            total_checks += 1
        
        return score / total_checks if total_checks > 0 else 0.0
    
    def analyze_model_compatibility(self, our_model: Dict, openrouter_models: List[Dict]) -> ModelCompatibility:
        """Analyze compatibility for a single model"""
        model_id = our_model["model_id"]
        openrouter_model = self.find_model_in_openrouter(model_id, openrouter_models)
        
        if not openrouter_model:
            return ModelCompatibility(
                model_id=model_id,
                name=our_model["model_id"],
                context_length=our_model.get("token_limit", 0),
                supported_parameters=[],
                pricing={},
                architecture={},
                is_available=False,
                compatibility_score=0.0,
                missing_features=self.our_expected_params.copy(),
                extra_features=[]
            )
        
        # Calculate compatibility
        compatibility_score = self.calculate_compatibility_score(our_model, openrouter_model)
        
        # Find missing and extra features
        supported_params = openrouter_model.get("supported_parameters", [])
        missing_features = [param for param in self.our_expected_params if param not in supported_params]
        extra_features = [param for param in supported_params if param not in self.our_expected_params]
        
        return ModelCompatibility(
            model_id=model_id,
            name=openrouter_model.get("name", model_id),
            context_length=openrouter_model.get("context_length", 0),
            supported_parameters=supported_params,
            pricing=openrouter_model.get("pricing", {}),
            architecture=openrouter_model.get("architecture", {}),
            is_available=True,
            compatibility_score=compatibility_score,
            missing_features=missing_features,
            extra_features=extra_features
        )
    
    def check_all_models(self) -> List[ModelCompatibility]:
        """Check compatibility for all our models"""
        print("Fetching OpenRouter models...")
        openrouter_data = self.fetch_openrouter_models()
        openrouter_models = openrouter_data.get("data", [])
        
        print(f"Found {len(openrouter_models)} models on OpenRouter")
        print(f"Checking compatibility for {len(self.our_models)} of our models...")
        
        compatibilities = []
        for our_model in self.our_models:
            compatibility = self.analyze_model_compatibility(our_model, openrouter_models)
            compatibilities.append(compatibility)
        
        return compatibilities
    
    def generate_report(self, compatibilities: List[ModelCompatibility]) -> str:
        """Generate a detailed compatibility report"""
        report = []
        report.append("# OpenRouter API Compatibility Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_models = len(compatibilities)
        available_models = sum(1 for c in compatibilities if c.is_available)
        high_compatibility = sum(1 for c in compatibilities if c.compatibility_score >= 0.8)
        medium_compatibility = sum(1 for c in compatibilities if 0.5 <= c.compatibility_score < 0.8)
        low_compatibility = sum(1 for c in compatibilities if c.compatibility_score < 0.5)
        
        report.append("## Summary")
        report.append(f"- Total models checked: {total_models}")
        report.append(f"- Available on OpenRouter: {available_models}")
        report.append(f"- High compatibility (≥80%): {high_compatibility}")
        report.append(f"- Medium compatibility (50-79%): {medium_compatibility}")
        report.append(f"- Low compatibility (<50%): {low_compatibility}")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        for compatibility in sorted(compatibilities, key=lambda x: x.compatibility_score, reverse=True):
            report.append(f"### {compatibility.model_id}")
            report.append(f"**Name:** {compatibility.name}")
            report.append(f"**Available:** {'✅ Yes' if compatibility.is_available else '❌ No'}")
            report.append(f"**Compatibility Score:** {compatibility.compatibility_score:.2%}")
            report.append(f"**Context Length:** {compatibility.context_length:,} tokens")
            
            if compatibility.is_available:
                report.append(f"**Supported Parameters:** {', '.join(compatibility.supported_parameters)}")
                
                if compatibility.missing_features:
                    report.append(f"**Missing Features:** {', '.join(compatibility.missing_features)}")
                
                if compatibility.extra_features:
                    report.append(f"**Extra Features:** {', '.join(compatibility.extra_features)}")
                
                # Pricing info
                pricing = compatibility.pricing
                if pricing:
                    report.append("**Pricing:**")
                    for key, value in pricing.items():
                        if value != "0":
                            report.append(f"  - {key}: ${value}")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        high_compat_models = [c for c in compatibilities if c.compatibility_score >= 0.8 and c.is_available]
        if high_compat_models:
            report.append("### High Compatibility Models (Recommended)")
            for model in high_compat_models:
                report.append(f"- **{model.model_id}**: {model.compatibility_score:.2%} compatibility")
            report.append("")
        
        missing_tool_support = [c for c in compatibilities if c.is_available and "tools" not in c.supported_parameters and c.compatibility_score > 0]
        if missing_tool_support:
            report.append("### Models Missing Tool Support")
            for model in missing_tool_support:
                report.append(f"- **{model.model_id}**: Consider disabling tool support or finding alternative")
            report.append("")
        
        unavailable_models = [c for c in compatibilities if not c.is_available]
        if unavailable_models:
            report.append("### Unavailable Models")
            for model in unavailable_models:
                report.append(f"- **{model.model_id}**: Not found on OpenRouter")
            report.append("")
        
        return "\n".join(report)
    
    def save_report(self, report: str, filename: str = None):
        """Save the report to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openrouter_compatibility_report_{timestamp}.md"
        
        # Save to docs folder
        docs_dir = os.path.join(project_root, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {filepath}")
        return filepath


def main():
    """Main function"""
    print("OpenRouter API Compatibility Checker")
    print("=" * 40)
    
    checker = OpenRouterCompatibilityChecker()
    
    print(f"Found {len(checker.our_models)} models in our configuration:")
    for model in checker.our_models:
        print(f"  - {model['provider_name']}: {model['model_id']}")
    print()
    
    # Check compatibility
    compatibilities = checker.check_all_models()
    
    # Generate and display report
    report = checker.generate_report(compatibilities)
    print(report)
    
    # Save report
    filepath = checker.save_report(report)
    
    print(f"\nCompatibility check complete! Report saved to: {filepath}")


if __name__ == "__main__":
    main()

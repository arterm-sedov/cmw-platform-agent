#!/usr/bin/env python3
"""
Cross-platform virtual environment setup and dependency installation for arterm-sedov.
Supports both Windows and Linux/macOS environments.

This script:
1. Creates a virtual environment
2. Installs dependencies using platform-specific requirements files
3. Handles platform-specific issues automatically
4. Provides comprehensive error handling and user feedback

Usage:
    python setup_venv.py [--skip-venv] [--skip-deps] [--verbose]
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import argparse

def print_status(message, status="INFO"):
    """Print a formatted status message."""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    
    if platform.system() == "Windows" and not os.environ.get("TERM"):
        # Windows without color support
        print(f"[{status}] {message}")
    else:
        # Unix-like systems or Windows with color support
        color = colors.get(status, colors["INFO"])
        reset = colors["RESET"]
        print(f"{color}[{status}]{reset} {message}")

def run_command(command, check=True, capture_output=True, shell=False):
    """
    Run a command and return the result.
    
    Args:
        command: Command to run (list or string)
        check: Whether to raise exception on non-zero exit code
        capture_output: Whether to capture stdout/stderr
        shell: Whether to run in shell mode
    
    Returns:
        subprocess.CompletedProcess object
    """
    try:
        if isinstance(command, str) and not shell:
            command = command.split()
        
        result = subprocess.run(
            command,
            check=check,
            capture_output=capture_output,
            shell=shell,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print_status(f"Command failed: {' '.join(command) if isinstance(command, list) else command}", "ERROR")
        print_status(f"Exit code: {e.returncode}", "ERROR")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        raise

def get_python_command():
    """Get the appropriate python command for the current platform."""
    if platform.system() == "Windows":
        return "python"
    else:
        return "python3"

def check_python_version():
    """Check if Python version is compatible (3.8+)."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status("Python 3.8+ is required", "ERROR")
        print_status(f"Current version: {version.major}.{version.minor}.{version.micro}", "ERROR")
        return False
    
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}", "SUCCESS")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_status("Virtual environment already exists", "WARNING")
        response = input("Do you want to recreate it? (y/N): ").strip().lower()
        if response != 'y':
            print_status("Using existing virtual environment", "INFO")
            return True
        else:
            print_status("Removing existing virtual environment...", "INFO")
            shutil.rmtree(venv_path)
    
    print_status("Creating virtual environment...", "INFO")
    python_cmd = get_python_command()
    
    try:
        run_command([python_cmd, "-m", "venv", "venv"])
        print_status("Virtual environment created successfully", "SUCCESS")
        return True
    except subprocess.CalledProcessError:
        print_status("Failed to create virtual environment", "ERROR")
        return False

def get_activation_command():
    """Get the activation command for the current platform."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def get_python_path():
    """Get the path to the virtual environment's Python executable."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\python.exe"
    else:
        return "venv/bin/python"

def get_pip_path():
    """Get the path to the virtual environment's pip executable."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip.exe"
    else:
        return "venv/bin/pip"

def get_requirements_file():
    """Get the appropriate requirements file based on the platform."""
    if platform.system() == "Windows":
        requirements_file = "requirements.win.txt"
        if Path(requirements_file).exists():
            print_status(f"Using Windows-specific requirements: {requirements_file}", "INFO")
            return requirements_file
        else:
            print_status("Windows requirements file not found, using main requirements.txt", "WARNING")
            return "requirements.txt"
    else:
        print_status("Using main requirements.txt for Linux/macOS", "INFO")
        return "requirements.txt"

def install_dependencies():
    """Install dependencies using the appropriate requirements file."""
    pip_cmd = get_pip_path()
    python_cmd = get_python_path()
    requirements_file = get_requirements_file()
    
    print_status("Installing dependencies...", "INFO")
    
    # Check if requirements file exists
    if not Path(requirements_file).exists():
        print_status(f"Requirements file {requirements_file} not found", "ERROR")
        return False
    
    # Step 1: Upgrade pip using python -m pip
    print_status("Upgrading pip...", "INFO")
    try:
        run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip"])
        print_status("Pip upgraded successfully", "SUCCESS")
    except subprocess.CalledProcessError:
        print_status("Failed to upgrade pip, continuing...", "WARNING")
    
    # Step 2: Install build tools
    print_status("Installing build tools...", "INFO")
    try:
        run_command([pip_cmd, "install", "wheel", "setuptools"])
    except subprocess.CalledProcessError:
        print_status("Failed to install build tools, continuing...", "WARNING")
    
    # Step 3: Install dependencies from requirements file
    print_status(f"Installing dependencies from {requirements_file}...", "INFO")
    try:
        run_command([pip_cmd, "install", "-r", requirements_file])
        print_status("All dependencies installed successfully", "SUCCESS")
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install dependencies from {requirements_file}", "ERROR")
        
        # If Windows requirements failed, try main requirements as fallback
        if platform.system() == "Windows" and requirements_file == "requirements.win.txt":
            print_status("Trying main requirements.txt as fallback...", "WARNING")
            try:
                run_command([pip_cmd, "install", "-r", "requirements.txt"])
                print_status("Dependencies installed using main requirements.txt", "SUCCESS")
                print_status("Note: TensorFlow not installed - sentence-transformers may not work optimally", "WARNING")
                print_status("To install TensorFlow manually, try:", "INFO")
                print_status("  pip install tensorflow-cpu", "INFO")
                print_status("  or", "INFO")
                print_status("  pip install tensorflow", "INFO")
                return True
            except subprocess.CalledProcessError:
                print_status("Both requirements files failed", "ERROR")
                return False
        
        return False

def verify_installation():
    """Verify that the installation was successful."""
    print_status("Verifying installation...", "INFO")
    
    python_cmd = get_python_path()
    
    # Test imports
    test_imports = [
        "numpy",
        "pandas", 
        "requests",
        "google.genai",
        "langchain",
        "supabase",
        "gradio"
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            run_command([python_cmd, "-c", f"import {module}"], capture_output=True)
            print_status(f"✓ {module}", "SUCCESS")
        except subprocess.CalledProcessError:
            print_status(f"✗ {module}", "ERROR")
            failed_imports.append(module)
    
    if failed_imports:
        print_status(f"Failed to import: {', '.join(failed_imports)}", "ERROR")
        return False
    
    # Test version info
    try:
        result = run_command([python_cmd, "-c", "import pandas as pd; print(f'Pandas version: {pd.__version__}')"], capture_output=True)
        print_status(result.stdout.strip(), "INFO")
    except subprocess.CalledProcessError:
        print_status("Could not get pandas version", "WARNING")
    
    print_status("Installation verification completed", "SUCCESS")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup virtual environment and install dependencies")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print_status("=" * 60, "INFO")
    print_status("arterm-sedov Setup Script", "INFO")
    print_status("=" * 60, "INFO")
    print_status(f"Platform: {platform.system()} {platform.release()}", "INFO")
    print_status(f"Python: {sys.executable}", "INFO")
    print_status("=" * 60, "INFO")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not args.skip_venv:
        if not create_virtual_environment():
            sys.exit(1)
    else:
        print_status("Skipping virtual environment creation", "INFO")
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            sys.exit(1)
    else:
        print_status("Skipping dependency installation", "INFO")
    
    # Verify installation
    if not args.skip_deps:
        if not verify_installation():
            print_status("Installation verification failed", "ERROR")
            sys.exit(1)
    
    # Print next steps
    print_status("=" * 60, "INFO")
    print_status("Setup completed successfully!", "SUCCESS")
    print_status("=" * 60, "INFO")
    print_status("Next steps:", "INFO")
    print_status("1. Activate the virtual environment:", "INFO")
    print_status(f"   {get_activation_command()}", "INFO")
    print_status("2. Set up your environment variables in .env file:", "INFO")
    print_status("   GEMINI_KEY=your_gemini_api_key", "INFO")
    print_status("   SUPABASE_URL=your_supabase_url", "INFO")
    print_status("   SUPABASE_KEY=your_supabase_key", "INFO")
    print_status("3. Run the agent:", "INFO")
    print_status("   python app.py", "INFO")
    print_status("=" * 60, "INFO")

if __name__ == "__main__":
    main() 
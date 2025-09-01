"""
Login Manager for CMW Platform Agent
===================================

This module handles all HuggingFace login operations for the CMW Platform Agent.
It provides a centralized way to manage login functionality and can be easily
enabled/disabled via configuration.

Usage:
    from login_manager import login_manager
    
    # Check if user is logged in
    if login_manager.is_logged_in(profile):
        username = login_manager.get_username(profile)
    else:
        login_message = login_manager.get_login_message()

Environment Variables:
    - LOGIN_ENABLED: Set to "true" to enable HuggingFace login (default: disabled)
"""

import os
from typing import Optional, Any

# Global flag to enable/disable login functionality
LOGIN_ENABLED = os.getenv("LOGIN_ENABLED", "false").lower() == "true"


class LoginManager:
    """
    Manages HuggingFace login operations for the CMW Platform Agent.
    
    This class provides a centralized interface for handling login functionality,
    particularly for Gradio applications. It can be easily disabled by setting
    the LOGIN_ENABLED environment variable to "false".
    """
    
    def __init__(self):
        """Initialize the login manager."""
        self.enabled = LOGIN_ENABLED
        if not self.enabled:
            print("ℹ️ HuggingFace login is disabled (set LOGIN_ENABLED=true to enable)")
    
    def get_status(self) -> dict:
        """Get the current status of the login manager."""
        return {
            "enabled": self.enabled,
            "login_enabled": LOGIN_ENABLED
        }
    
    def is_logged_in(self, profile: Optional[Any]) -> bool:
        """
        Check if user is logged in to HuggingFace.
        
        Args:
            profile: Gradio OAuth profile object or None
            
        Returns:
            bool: True if user is logged in, False otherwise
        """
        if not self.enabled:
            return False
        
        return profile is not None and hasattr(profile, 'username') and profile.username
    
    def get_username(self, profile: Optional[Any]) -> Optional[str]:
        """
        Get the username from the profile.
        
        Args:
            profile: Gradio OAuth profile object or None
            
        Returns:
            str: Username if logged in, None otherwise
        """
        if not self.enabled:
            return None
        
        if self.is_logged_in(profile):
            return f"{profile.username}"
        return None
    
    def get_login_message(self) -> str:
        """
        Get the login message to display to users.
        
        Returns:
            str: Login message
        """
        if not self.enabled:
            return "HuggingFace login is disabled. Running in local mode."
        
        return "Please Login to Hugging Face with the button."
    
    def validate_login_for_operation(self, profile: Optional[Any], operation_name: str = "operation") -> tuple[bool, Optional[str], Optional[str]]:
        """
        Validate login for a specific operation.
        
        Args:
            profile: Gradio OAuth profile object or None
            operation_name: Name of the operation requiring login
            
        Returns:
            tuple: (is_valid, username, error_message)
        """
        if not self.enabled:
            return True, None, None
        
        if self.is_logged_in(profile):
            username = self.get_username(profile)
            print(f"User logged in: {username}")
            return True, username, None
        else:
            print("User not logged in.")
            error_message = f"Login required for {operation_name}. {self.get_login_message()}"
            return False, None, error_message
    
    def get_login_button(self):
        """
        Get the Gradio login button component.
        
        Returns:
            Gradio component: Login button or None if disabled
        """
        if not self.enabled:
            return None
        
        try:
            import gradio as gr
            return gr.LoginButton()
        except ImportError:
            print("Warning: gradio not available for login button")
            return None


# Global login manager instance
login_manager = LoginManager()

# Convenience functions for backward compatibility
def is_logged_in(profile: Optional[Any]) -> bool:
    """Check if user is logged in to HuggingFace."""
    return login_manager.is_logged_in(profile)

def get_username(profile: Optional[Any]) -> Optional[str]:
    """Get the username from the profile."""
    return login_manager.get_username(profile)

def get_login_message() -> str:
    """Get the login message to display to users."""
    return login_manager.get_login_message()

def validate_login_for_operation(profile: Optional[Any], operation_name: str = "operation") -> tuple[bool, Optional[str], Optional[str]]:
    """Validate login for a specific operation."""
    return login_manager.validate_login_for_operation(profile, operation_name)

def get_login_button():
    """Get the Gradio login button component."""
    return login_manager.get_login_button()

def get_login_manager_status() -> dict:
    """Get login manager status."""
    return login_manager.get_status()

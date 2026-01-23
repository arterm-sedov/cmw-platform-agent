"""
Dead Code: Debug/Utility Functions
=================================

These debug and utility functions were defined but never called anywhere in the codebase.
They can be safely removed unless needed for future debugging.

Extracted from:
- agent_ng/response_processor.py
- agent_ng/message_processor.py
- agent_ng/debug_streamer.py
- agent_ng/trace_manager.py
"""

# From agent_ng/response_processor.py
def print_message_components(self, msg: Any, msg_index: int = 0):
    """Print detailed message components for debugging"""
    print(f"\\n=== Message {msg_index} Components ===")

    if hasattr(msg, '__dict__'):
        for attr, value in msg.__dict__.items():
            if not attr.startswith('_'):
                self._print_attribute(attr, value)
    else:
        # For non-object types, print basic info
        print(f"Type: {type(msg)}")
        print(f"Value: {msg}")

    # Check for common LangChain message attributes
    common_attrs = ['content', 'type', 'additional_kwargs', 'response_metadata', 'id', 'name', 'tool_calls']
    for attr in common_attrs:
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            self._print_attribute(attr, value)

def _print_attribute(self, attr_name: str, value: Any):
    """Print a single attribute with proper formatting"""
    if isinstance(value, (str, int, float, bool, type(None))):
        print(f"  {attr_name}: {value}")
    elif isinstance(value, (list, tuple)):
        print(f"  {attr_name}: {type(value).__name__} with {len(value)} items")
        for i, item in enumerate(value[:3]):  # Show first 3 items
            print(f"    [{i}]: {type(item).__name__} - {str(item)[:100]}...")
        if len(value) > 3:
            print(f"    ... and {len(value) - 3} more items")
    elif isinstance(value, dict):
        print(f"  {attr_name}: dict with {len(value)} keys")
        for key, val in list(value.items())[:3]:  # Show first 3 key-value pairs
            print(f"    {key}: {type(val).__name__} - {str(val)[:100]}...")
        if len(value) > 3:
            print(f"    ... and {len(value) - 3} more keys")
    else:
        print(f"  {attr_name}: {type(value).__name__} - {str(value)[:100]}...")

# From agent_ng/message_processor.py
def print_message_components(self, msg: Any, msg_index: int = 0):
    """Print detailed message components for debugging"""
    print(f"\\n=== Message {msg_index} Components ===")

    if hasattr(msg, '__dict__'):
        for attr, value in msg.__dict__.items():
            if not attr.startswith('_'):
                self._print_attribute(attr, value)
    else:
        # For non-object types, print basic info
        print(f"Type: {type(msg)}")
        print(f"Value: {msg}")

    # Check for common LangChain message attributes
    common_attrs = ['content', 'type', 'additional_kwargs', 'response_metadata', 'id', 'name', 'tool_calls']
    for attr in common_attrs:
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            self._print_attribute(attr, value)

def _print_attribute(self, attr_name: str, value: Any):
    """Print a single attribute with proper formatting"""
    if isinstance(value, (str, int, float, bool, type(None))):
        print(f"  {attr_name}: {value}")
    elif isinstance(value, (list, tuple)):
        print(f"  {attr_name}: {type(value).__name__} with {len(value)} items")
        for i, item in enumerate(value[:3]):  # Show first 3 items
            print(f"    [{i}]: {type(item).__name__} - {str(item)[:100]}...")
        if len(value) > 3:
            print(f"    ... and {len(value) - 3} more items")
    elif isinstance(value, dict):
        print(f"  {attr_name}: dict with {len(value)} keys")
        for key, val in list(value.items())[:3]:  # Show first 3 key-value pairs
            print(f"    {key}: {type(val).__name__} - {str(val)[:100]}...")
        if len(value) > 3:
            print(f"    ... and {len(value) - 3} more keys")
    else:
        print(f"  {attr_name}: {type(value).__name__} - {str(value)[:100]}...")

# From agent_ng/debug_streamer.py
def get_recent_logs(self, count: int = 50) -> list[LogEntry]:
    """Get recent log entries (for debugging)"""
    # This is a simple implementation - in production you might want to use a proper log store
    return []

# From agent_ng/trace_manager.py
def trace_prints_with_context(context_type: str):
    """Decorator to trace print statements with context"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Store original print
            original_print = print

            def trace_print(*print_args, **print_kwargs):
                # Original print functionality
                original_print(*print_args, **print_kwargs)

                # Add to trace
                if hasattr(self, 'trace_manager'):
                    message = ' '.join(str(arg) for arg in print_args)
                    self.trace_manager.add_debug_output(f"[{context_type}] {message}", context_type)

            # Replace print temporarily
            import builtins
            builtins.print = trace_print

            try:
                result = func(self, *args, **kwargs)
            finally:
                # Restore original print
                builtins.print = original_print

            return result
        return wrapper
    return decorator

def trace_prints(func):
    """Decorator to trace print statements"""
    def wrapper(self, *args, **kwargs):
        # Store original print
        original_print = print

        def trace_print(*print_args, **print_kwargs):
            # Original print functionality
            original_print(*print_args, **print_kwargs)

            # Add to trace
            if hasattr(self, 'trace_manager'):
                message = ' '.join(str(arg) for arg in print_args)
                self.trace_manager.add_debug_output(message, "general")

        # Replace print temporarily
        import builtins
        builtins.print = trace_print

        try:
            result = func(self, *args, **kwargs)
        finally:
            # Restore original print
            builtins.print = original_print

        return result
    return wrapper

class Tee:
    """
    Tee class to duplicate writes to multiple streams (e.g., sys.stdout and a buffer).
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

class _SinkWriter:
    """Writer that sends data to a sink function"""
    def __init__(self, sink):
        self.sink = sink

    def write(self, data):
        self.sink(data)

    def flush(self):
        pass

def get_trace_history(self) -> List[dict]:
    """Get trace history"""
    return [self._serialize_trace_data(trace) for trace in self.trace_history]

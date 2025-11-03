"""Base tool classes that can be imported without vllm dependencies."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

@dataclass
class ToolOutput:
    output: str
    called: bool
    error: str
    timeout: bool
    runtime: float
    start_str: str = "<output>\n"
    end_str: str = "\n</output>"


@dataclass
class ToolCallInfo:
    """Structured information about a detected tool call.
    
    This is returned by is_triggered() when a tool call is detected, containing
    the extracted input, parsed arguments, and metadata that can be used directly in tool execution.
    
    The parser can extract structured arguments beyond just the raw text content,
    enabling tools to receive pre-parsed parameters (e.g., from JSON, function calls, etc.).
    """
    # Whether the tool is actually triggered
    triggered: bool
    # The extracted input content (e.g., code, query, URL) - raw text from between tags
    input_content: str = ""
    # Structured arguments parsed from the tool call (e.g., JSON, named parameters)
    # This allows tools to receive pre-parsed structured data instead of parsing themselves
    arguments: Dict[str, Any] = None
    
    # Start position of the tool call in the original prompt (for debugging/validation)
    start_pos: Optional[int] = None
    # End position of the tool call in the original prompt (for debugging/validation)
    end_pos: Optional[int] = None
    # The full prompt text (for tools that need the full context)
    full_prompt: str = ""
    
    def __post_init__(self):
        """Ensure arguments is a dict if not provided."""
        if self.arguments is None:
            self.arguments = {}
    
    def __bool__(self) -> bool:
        """Allow truthiness check: if ToolCallInfo is triggered."""
        return self.triggered


class Tool:
    """Base class for all tools with configurable trigger detection and vLLM compatibility.
    
    Tools can control when they are triggered by:
    1. Using the default behavior: triggered when text ends with `end_str`
    2. Overriding `is_triggered()` for custom detection logic (can return ToolCallInfo for efficiency)
    3. Providing vLLM function calling format for native vLLM tool calling
    
    The `is_triggered()` method can return either:
    - bool: Simple trigger check (backward compatible)
    - ToolCallInfo: Structured object with extracted input (more efficient, avoids double parsing)
    
    If ToolCallInfo is returned with triggered=True, the `input_content` field will be passed
    to `__call__()` instead of re-parsing the prompt.
    """
    
    def __init__(
        self,
        name: str,
        start_str: str = "",
        end_str: str = "",
        *args,
        **kwargs
    ):
        """
        Args:
            name: Unique name for the tool
            start_str: Starting delimiter for tool calls (e.g., "<code>")
            end_str: Ending delimiter for tool calls (e.g., "</code>"). Used as stop string.
        """
        self.name = name
        self.start_str = start_str
        self.end_str = end_str

    def get_name(self) -> str:
        return self.name

    def __call__(self, prompt_or_info: Union[str, ToolCallInfo]) -> ToolOutput:
        """Execute the tool with the given prompt or ToolCallInfo.
        
        If a ToolCallInfo is passed (from is_triggered), use:
        - `input_content`: pre-extracted raw text from between tags
        - `arguments`: pre-parsed structured arguments (if available)
        
        Otherwise, parse the prompt string to extract input between start_str and end_str.
        
        Args:
            prompt_or_info: Either the full prompt string, or a ToolCallInfo object
                from is_triggered() that contains pre-extracted input and optionally
                pre-parsed arguments.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def is_triggered(self, prompt: str) -> Union[bool, ToolCallInfo]:
        """Check if this tool should be triggered and optionally extract input and arguments.
        
        Can return either:
        - bool: Simple trigger check (backward compatible, default behavior)
        - ToolCallInfo: Structured object with extracted input and parsed arguments (more efficient)
        
        If returning ToolCallInfo with triggered=True:
        - Include the extracted `input_content` (raw text between tags)
        - Optionally include `arguments` dict with structured parsed parameters
          (e.g., JSON-parsed data, function call parameters, etc.)
        - This avoids double parsing in __call__()
        
        If returning bool, __call__() will need to parse the prompt itself.
        
        Default behavior: checks if prompt ends with `end_str` (allowing trailing whitespace),
        and optionally verifies that `start_str` appears earlier in the text if both are set.
        Returns a ToolCallInfo with the extracted content between tags.
        
        For tools that need structured arguments (e.g., function calls with parameters),
        override this method to parse the input_content into an arguments dict.
        
        Example for a tool with JSON arguments:
            ```python
            def is_triggered(self, prompt: str) -> Union[bool, ToolCallInfo]:
                info = super().is_triggered(prompt)
                if isinstance(info, ToolCallInfo) and info.triggered:
                    try:
                        info.arguments = json.loads(info.input_content)
                    except json.JSONDecodeError:
                        # Fall back to treating input_content as plain text
                        info.arguments = {"text": info.input_content}
                return info
            ```
        
        Args:
            prompt: The generated text or prompt to check
            
        Returns:
            - False if not triggered
            - True or ToolCallInfo with triggered=True if triggered (ToolCallInfo is preferred for efficiency)
        """
        if not self.end_str:
            return False
        
        # Strip trailing whitespace for more flexible matching
        # (models may generate trailing newlines/spaces before the closing tag)
        prompt_stripped = prompt.rstrip()
        
        # Check if prompt ends with end_str (after stripping whitespace)
        if not prompt_stripped.endswith(self.end_str):
            return False
        
        # If start_str is also set, verify it appears earlier in the text and extract content
        if self.start_str:
            # Find the position where end_str starts
            end_pos = len(prompt_stripped) - len(self.end_str)
            
            # Check if start_str appears before the end_str
            # Allow for content between start_str and end_str
            start_pos = prompt_stripped.rfind(self.start_str, 0, end_pos)
            
            if start_pos == -1:
                # start_str not found before end_str - not a valid trigger
                return False
            
            # Extract content between the tags
            content_start = start_pos + len(self.start_str)
            input_content = prompt_stripped[content_start:end_pos].strip()
            
            # Return ToolCallInfo with extracted content for efficiency
            return ToolCallInfo(
                triggered=True,
                input_content=input_content,
                start_pos=start_pos,
                end_pos=len(prompt_stripped),
                full_prompt=prompt
            )
        
        # No start_str, just end_str at the end - return simple True for backward compat
        # But ideally tools should define start_str for better detection
        return True

    def get_stop_strings(self) -> List[str]:
        """Get the stop strings that should trigger this tool.
        
        These are typically used in SamplingParams.stop to stop generation
        when the model outputs a tool call delimiter.
        
        Returns:
            List of stop strings. Defaults to [end_str] if end_str is set.
        """
        if self.end_str:
            return [self.end_str]
        return []

    def to_vllm_function_format(self) -> Optional[Dict[str, Any]]:
        """Convert this tool to vLLM's function calling format.
        
        vLLM supports function calling via JSON schema. This method allows tools
        to optionally define themselves in that format for native vLLM integration.
        
        Returns:
            Dict with 'type', 'function' keys following vLLM's function calling format,
            or None if this tool doesn't support vLLM function calling format.
            
        Example:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        """
        # Default implementation: return None (not vLLM function calling compatible)
        # Subclasses can override to provide vLLM format
        return None


class MaxCallsExceededTool(Tool):
    def __init__(self, start_str: str = "<tool>", end_str: str = "</tool>"):
        super().__init__("MaxCallsExceededTool", start_str=start_str, end_str=end_str)

    def __call__(self, prompt_or_info: Union[str, ToolCallInfo]) -> ToolOutput:
        return ToolOutput(
            output="Max tool calls exceeded.", called=False, error="Max tool calls exceeded", timeout=False, runtime=0
        )

    def is_triggered(self, prompt: str) -> Union[bool, ToolCallInfo]:
        # This tool is not triggered by any prompt.
        return False

    def get_stop_strings(self) -> List[str]:
        return []

from dataclasses import dataclass
from typing import Any

from open_instruct.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolOutput:
    output: str
    called: bool
    error: str
    timeout: bool
    runtime: float


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]

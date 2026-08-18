"""
Base Agent Class
================
Every agent in the OPC AI Video Studio inherits from this base class.
It provides common functionality: logging, config access, and a standard
execute interface that the orchestrator calls.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


class BaseAgent(ABC):
    """Base class for all AI Video Studio agents."""

    def __init__(self, name: str, role: str, config: Dict[str, Any]):
        self.name = name
        self.role = role
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"agent.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"[%(asctime)s] [{self.name.upper()}] %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def announce(self, message: str):
        """Print a styled announcement to the console."""
        console.print(Panel(
            escape(message),
            title=f"[bold cyan]{escape(self.name)}[/bold cyan] | {escape(self.role)}",
            border_style="cyan",
            padding=(0, 2),
        ))

    def log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(self.logger, level)(message)

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            input_data: Dictionary containing input from previous agents.

        Returns:
            Dictionary of results to pass to the next agent.
        """
        pass

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper around execute() that adds timing and error handling.
        Called by the orchestrator.
        """
        self.announce(f"Starting: {self.role}")
        start_time = time.time()

        try:
            result = self.execute(input_data)
            elapsed = time.time() - start_time
            self.announce(f"Completed in {elapsed:.1f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            self.announce(f"FAILED after {elapsed:.1f}s: {e}")
            raise

    def ensure_dir(self, path: str) -> Path:
        """Ensure a directory exists, create if needed."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

"""
AIR Logging Configuration

Provides configurable logging with colored console output and optional file logging.
Supports multiple log levels and customizable formatting.

Example:
    >>> from air.utils.logging import get_logger, setup_logging
    >>> setup_logging(level="DEBUG", colored=True)
    >>> logger = get_logger(__name__)
    >>> logger.info("AIR initialized successfully")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

# ANSI color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
    "BOLD": "\033[1m",  # Bold
    "DIM": "\033[2m",  # Dim
}

# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging has been set up
_logging_configured = False


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color codes to log messages for terminal output.

    Attributes:
        use_colors: Whether to apply ANSI color codes to output.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        """
        Initialize the colored formatter.

        Args:
            fmt: Log message format string.
            datefmt: Date format string.
            use_colors: Whether to use colored output.
        """
        super().__init__(fmt=fmt or DEFAULT_FORMAT, datefmt=datefmt or DEFAULT_DATE_FORMAT)
        self.use_colors = use_colors and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        """Check if the terminal supports color output."""
        # Check if stdout is a TTY
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check for NO_COLOR environment variable (standard)
        if os.environ.get("NO_COLOR"):
            return False

        # Check for FORCE_COLOR environment variable
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check terminal type
        term = os.environ.get("TERM", "")
        return term not in ("dumb", "")

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with optional color codes.

        Args:
            record: The log record to format.

        Returns:
            Formatted log message string.
        """
        if self.use_colors:
            # Get color for this level
            color = COLORS.get(record.levelname, COLORS["RESET"])
            reset = COLORS["RESET"]

            # Colorize the level name
            original_levelname = record.levelname
            record.levelname = f"{color}{record.levelname}{reset}"

            # Format the message
            formatted = super().format(record)

            # Restore original values
            record.levelname = original_levelname

            return formatted

        return super().format(record)


class AIRLogger(logging.Logger):
    """
    Custom logger class for AIR with additional functionality.

    Extends the standard logging.Logger with AIR-specific methods
    and default configuration.
    """

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        """
        Initialize the AIR logger.

        Args:
            name: Logger name.
            level: Logging level.
        """
        super().__init__(name, level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Creates a logger with AIR's default configuration if logging
    has not been explicitly configured.

    Args:
        name: The name for the logger, typically __name__.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing request")
    """
    global _logging_configured

    # Ensure basic logging is configured
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)


def setup_logging(
    level: Union[str, int] = "INFO",
    colored: bool = True,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Configure logging for the AIR package.

    Sets up console logging with optional colored output and optional
    file logging. Should be called once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int.
        colored: Whether to use colored output in console.
        log_file: Optional path to write logs to a file.
        log_format: Custom log format string.
        date_format: Custom date format string.

    Example:
        >>> setup_logging(level="DEBUG", colored=True)
        >>> setup_logging(level="INFO", log_file="/var/log/air.log")
    """
    global _logging_configured

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get the root AIR logger
    root_logger = logging.getLogger("air")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        fmt=log_format,
        datefmt=date_format,
        use_colors=colored,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_path = Path(log_file)

        # Create parent directories if they don't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)

        # File output should not be colored
        file_formatter = logging.Formatter(
            fmt=log_format or DEFAULT_FORMAT,
            datefmt=date_format or DEFAULT_DATE_FORMAT,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    _logging_configured = True


def set_level(level: Union[str, int]) -> None:
    """
    Change the logging level for all AIR loggers.

    Args:
        level: New logging level.

    Example:
        >>> set_level("DEBUG")  # Enable debug output
        >>> set_level(logging.WARNING)  # Only warnings and above
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("air")
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        handler.setLevel(level)


def add_file_handler(
    log_file: Union[str, Path],
    level: Optional[Union[str, int]] = None,
    log_format: Optional[str] = None,
) -> logging.FileHandler:
    """
    Add a file handler to the AIR logger.

    Args:
        log_file: Path to the log file.
        level: Optional logging level for this handler.
        log_format: Optional custom format for this handler.

    Returns:
        The created FileHandler instance.

    Example:
        >>> add_file_handler("/var/log/air_debug.log", level="DEBUG")
    """
    root_logger = logging.getLogger("air")
    log_path = Path(log_file)

    # Create parent directories if they don't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        file_handler.setLevel(level)
    else:
        file_handler.setLevel(root_logger.level)

    file_formatter = logging.Formatter(
        fmt=log_format or DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(file_handler)

    return file_handler


def create_rotating_file_handler(
    log_file: Union[str, Path],
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    level: Optional[Union[str, int]] = None,
) -> logging.Handler:
    """
    Create a rotating file handler that archives old logs.

    Args:
        log_file: Path to the log file.
        max_bytes: Maximum file size before rotation (default 10 MB).
        backup_count: Number of backup files to keep.
        level: Optional logging level for this handler.

    Returns:
        The created RotatingFileHandler instance.

    Example:
        >>> create_rotating_file_handler("/var/log/air.log", max_bytes=5*1024*1024)
    """
    from logging.handlers import RotatingFileHandler

    root_logger = logging.getLogger("air")
    log_path = Path(log_file)

    # Create parent directories if they don't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        handler.setLevel(level)
    else:
        handler.setLevel(root_logger.level)

    formatter = logging.Formatter(
        fmt=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    return handler

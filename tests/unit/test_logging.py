"""
Unit tests for AIR logging utilities.
"""

import logging
import tempfile
from pathlib import Path

from air.utils.logging import (
    ColoredFormatter,
    add_file_handler,
    get_logger,
    set_level,
    setup_logging,
)


class TestColoredFormatter:
    """Tests for the ColoredFormatter class."""

    def test_format_without_colors(self):
        """Test formatting without color codes."""
        formatter = ColoredFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "INFO" in formatted

    def test_format_with_colors_disabled_via_env(self, monkeypatch):
        """Test that NO_COLOR environment variable disables colors."""
        monkeypatch.setenv("NO_COLOR", "1")
        formatter = ColoredFormatter(use_colors=True)
        # The formatter should detect NO_COLOR and disable colors
        assert not formatter.use_colors


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_logger_instance(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_multiple_calls_return_same_logger(self):
        """Test that multiple calls return the same logger."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_sets_log_level(self):
        """Test that setup_logging sets the correct log level."""
        setup_logging(level="DEBUG")
        root_logger = logging.getLogger("air")
        assert root_logger.level == logging.DEBUG

    def test_string_level_conversion(self):
        """Test that string levels are properly converted."""
        setup_logging(level="WARNING")
        root_logger = logging.getLogger("air")
        assert root_logger.level == logging.WARNING

    def test_file_logging(self):
        """Test that file logging can be configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(level="INFO", log_file=log_file)

            logger = get_logger("air.test")
            logger.info("Test file logging message")

            # Verify the log file was created
            assert log_file.exists()


class TestSetLevel:
    """Tests for the set_level function."""

    def test_change_level_string(self):
        """Test changing level with string."""
        setup_logging(level="INFO")
        set_level("ERROR")
        root_logger = logging.getLogger("air")
        assert root_logger.level == logging.ERROR

    def test_change_level_int(self):
        """Test changing level with int."""
        setup_logging(level="INFO")
        set_level(logging.CRITICAL)
        root_logger = logging.getLogger("air")
        assert root_logger.level == logging.CRITICAL


class TestAddFileHandler:
    """Tests for the add_file_handler function."""

    def test_creates_log_file(self):
        """Test that add_file_handler creates the log file."""
        setup_logging(level="INFO")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "additional.log"
            handler = add_file_handler(log_file)

            assert isinstance(handler, logging.FileHandler)

            # Write a log message
            logger = get_logger("air.test")
            logger.info("Test additional file logging")

            # Verify the file exists
            assert log_file.exists()

    def test_creates_parent_directories(self):
        """Test that add_file_handler creates parent directories."""
        setup_logging(level="INFO")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "nested" / "test.log"
            add_file_handler(log_file)

            logger = get_logger("air.test")
            logger.info("Test nested directory logging")

            assert log_file.exists()

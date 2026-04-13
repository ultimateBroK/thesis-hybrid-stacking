"""Tests for ColoredFormatter ANSI isolation.

Verifies that ANSI escape codes do NOT bleed into file handler output
while colored output IS produced for console handler.
"""

import logging

import main


def test_colored_formatter_no_ansi_in_output():
    """ColoredFormatter must not leave ANSI codes in formatted text after restore."""
    formatter = main.ColoredFormatter("%(levelname)s | %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )

    original = record.levelname  # "INFO"
    result = formatter.format(record)

    # After formatting, the original levelname should be restored
    assert record.levelname == original
    # The result should contain ANSI codes (for console)
    assert "\033[" in result
    # But the record's levelname must NOT contain ANSI after the call
    assert "\033[" not in record.levelname


def test_two_handlers_independent(tmp_path):
    """Console handler gets color, file handler gets plain text."""
    log_path = tmp_path / "test.log"

    # Set up a logger with ColoredFormatter for console and plain for file
    logger = logging.getLogger(f"test_ansi_{id(log_path)}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler (colored)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(main.ColoredFormatter("%(levelname)s %(message)s"))
    logger.addHandler(console_handler)

    # File handler (plain)
    file_handler = logging.FileHandler(str(log_path), mode="w")
    file_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    logger.info("test message")

    # Flush and close
    for h in logger.handlers:
        h.flush()
        h.close()
    logger.handlers.clear()

    # File should NOT contain ANSI
    log_text = log_path.read_text()
    assert "\033[" not in log_text
    assert "INFO" in log_text
    assert "test message" in log_text

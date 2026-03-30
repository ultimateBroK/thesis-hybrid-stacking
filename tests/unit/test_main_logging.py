"""Tests for pipeline logging setup."""

import io
import logging
import sys
from pathlib import Path

import main


def test_setup_logging_records_stdout_stderr_and_logger(tmp_path, monkeypatch):
    log_path = tmp_path / "pipeline.log"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    monkeypatch.setattr(main, "_ORIGINAL_STDOUT", stdout_buffer)
    monkeypatch.setattr(main, "_ORIGINAL_STDERR", stderr_buffer)
    monkeypatch.setattr(main, "_PIPELINE_LOG_STREAM", None)
    monkeypatch.setattr(sys, "stdout", stdout_buffer)
    monkeypatch.setattr(sys, "stderr", stderr_buffer)

    logger = main.setup_logging(log_path)

    try:
        print("stdout message")
        sys.stderr.write("stderr message\n")
        logger.info("logger message")
        sys.stdout.flush()
        sys.stderr.flush()
        for handler in logging.getLogger().handlers:
            handler.flush()
    finally:
        if main._PIPELINE_LOG_STREAM is not None and not main._PIPELINE_LOG_STREAM.closed:
            main._PIPELINE_LOG_STREAM.close()
        monkeypatch.setattr(sys, "stdout", stdout_buffer)
        monkeypatch.setattr(sys, "stderr", stderr_buffer)

    log_text = Path(log_path).read_text(encoding="utf-8")

    assert "stdout message" in log_text
    assert "stderr message" in log_text
    assert "logger message" in log_text

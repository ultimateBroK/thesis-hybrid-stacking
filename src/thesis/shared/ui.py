"""Shared lightweight CLI UI helpers without Rich dependency."""

from contextlib import contextmanager
import logging
import re

logger = logging.getLogger("thesis.ui")
_RICH_TAG_RE = re.compile(r"\[/?[^\]]+\]")


class SimpleConsole:
    """Minimal console facade compatible with previous Rich call sites."""

    def print(self, *args, **kwargs) -> None:
        """Log plain text messages."""
        message = " ".join(str(arg) for arg in args)
        message = _RICH_TAG_RE.sub("", message).strip()
        if message:
            logger.info(message)

    def rule(self, title: str | None = None, **kwargs) -> None:
        """Log a visual separator line."""
        if title:
            title = _RICH_TAG_RE.sub("", title).strip()
            logger.info("---- %s ----", title)
        else:
            logger.info("--------------------")

    @contextmanager
    def status(self, message: str):
        """Context manager that logs a status message once."""
        message = _RICH_TAG_RE.sub("", message).strip()
        logger.info("%s", message)
        yield


console = SimpleConsole()

# Stage colour map (used by pipeline + training)
STAGE_STYLES: dict[int, str] = {
    1: "bold blue",
    2: "bold green",
    3: "bold yellow",
    4: "bold cyan",
    5: "bold magenta",
    6: "bold red",
}

STAGE_LABELS: dict[int, str] = {
    1: "Data Preparation",
    2: "Feature Engineering",
    3: "Label Generation",
    4: "Model Training",
    5: "Application Demo / Backtest",
    6: "Report Generation",
}


# UI helpers
def stage_header(stage: int) -> None:
    """Print a stage banner with concise log output.

    Args:
        stage: Stage number (1-indexed, 1–6).
    """
    _logger = logging.getLogger("thesis")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    total = 6
    console.rule(f"STAGE {stage}/{total} | {label}")
    # Logger output for file capture
    _logger.info("STAGE %d/%d | %s", stage, total, label)


def stage_skip(stage: int, reason: str) -> None:
    """Print a skip line and logger message.

    Args:
        stage: Stage number (1-indexed, 1–6).
        reason: Why the stage is being skipped.
    """
    _logger = logging.getLogger("thesis")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    console.print(f"SKIP {label}: {reason}")
    # Logger output for file capture
    _logger.info("SKIP %s | %s", label, reason)

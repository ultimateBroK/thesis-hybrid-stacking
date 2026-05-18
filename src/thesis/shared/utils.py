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

# Stage colour map (used by pipeline + training) — 4 stages
STAGE_STYLES: dict[int, str] = {
    1: "bold blue",
    2: "bold green",
    3: "bold yellow",
    4: "bold cyan",
}

STAGE_LABELS: dict[int, str] = {
    1: "Market Data Preparation",
    2: "ML Dataset Construction",
    3: "Model Training & Evaluation",
    4: "Report Generation",
}


# UI helpers
def stage_header(stage: int) -> None:
    """Print a stage banner with concise log output."""
    _logger = logging.getLogger("thesis")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    total = 4
    console.rule(f"STAGE {stage}/{total} | {label}")
    _logger.info("STAGE %d/%d | %s", stage, total, label)


def stage_skip(stage: int, reason: str) -> None:
    """Print a skip line and logger message."""
    _logger = logging.getLogger("thesis")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    console.print(f"SKIP {label}: {reason}")
    _logger.info("SKIP %s | %s", label, reason)

"""Suppress same-document duplicate object warnings in autodoc builds.

When autodoc processes class-level attributes with docstrings, the Python domain
sometimes registers the same object twice within the same document, triggering
"duplicate object description" warnings.  This filter silences those warnings
when both registrations originate from the same document --- effectively treating
them as idempotent re-registrations rather than true conflicts.
"""

from __future__ import annotations

import logging
import re

_DUP_RE = re.compile(
    r"duplicate object description of .+, "
    r"other instance in (?P<other>[^,]+), "
    r"use :no-index: for one of them"
)


class SameDocumentDuplicateFilter(logging.Filter):
    """Drop duplicate-object warnings when both instances share a document."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        m = _DUP_RE.search(msg)
        if not m:
            return True

        other_doc = m.group("other")

        # Sphinx location can be a string, tuple, or docutils Node.
        # Extract the document name from whatever form it takes.
        location = getattr(record, "location", None)
        current_doc = ""

        if location is None:
            current_doc = ""
        elif isinstance(location, str):
            current_doc = (
                location.split(":")[0] if ":" in location else location
            )
        elif isinstance(location, tuple) and len(location) >= 1:
            current_doc = str(location[0]) if location[0] else ""
        else:
            # location is a docutils Node --- walk up to find the document
            try:
                node = location
                while node is not None:
                    doc = getattr(node, "document", None)
                    if doc is not None:
                        current_doc = getattr(
                            doc, "attributes", {}
                        ).get("source", "")
                        if not current_doc:
                            current_doc = getattr(doc, "source", "")
                        break
                    node = getattr(node, "parent", None)
            except Exception:
                current_doc = ""

        # Convert full filesystem path to Sphinx docname
        if "/docs/source/" in current_doc:
            current_doc = current_doc.split("/docs/source/", 1)[1]
        current_doc = current_doc.replace(".rst", "").replace(".md", "")

        if current_doc == other_doc:
            return False

        # Also check if base names match
        current_base = (
            current_doc.rsplit("/", 1)[-1]
            if "/" in current_doc
            else current_doc
        )
        other_base = (
            other_doc.rsplit("/", 1)[-1] if "/" in other_doc else other_doc
        )
        if current_base == other_base:
            return False

        return True


def setup(app):
    """Register the duplicate-object filter with the 'sphinx' logger."""
    logger = logging.getLogger("sphinx")
    logger.addFilter(SameDocumentDuplicateFilter())
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

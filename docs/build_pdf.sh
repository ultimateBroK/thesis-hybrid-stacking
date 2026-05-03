#!/usr/bin/env bash
# Build PDF documentation via Sphinx LaTeX builder.
# Gracefully handles missing LaTeX by printing instructions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/_build"

echo "==> Building LaTeX sources..."
sphinx-build -b latex -W "$SCRIPT_DIR/source" "$BUILD_DIR/latex"

LATEX_DIR="$BUILD_DIR/latex"
if [ ! -f "$LATEX_DIR"/*.tex ]; then
    echo "ERROR: No .tex file produced."
    exit 1
fi

if command -v pdflatex &>/dev/null; then
    echo "==> Compiling PDF with pdflatex..."
    (
        cd "$LATEX_DIR"
        # Two passes for TOC / cross-refs
        pdflatex -interaction=nonstopmode *.tex
        pdflatex -interaction=nonstopmode *.tex
    )
    echo "==> PDF built: $LATEX_DIR/thesis.pdf"
else
    echo "==> pdflatex not found. Skipping PDF compilation."
    echo "    Install texlive-latex-base (or equivalent) to build PDFs."
    echo "    LaTeX sources are ready at: $LATEX_DIR"
fi

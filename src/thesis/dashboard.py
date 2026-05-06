"""Streamlit launch shim — keeps ``pixi run streamlit`` working."""

from thesis.dashboard._main import main  # noqa: F401

main()

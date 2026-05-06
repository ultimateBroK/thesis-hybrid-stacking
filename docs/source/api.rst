API Reference
=============

The public documentation currently focuses on workflow guides. The codebase uses
stage-based modules under ``src/thesis/``:

- ``thesis._shared`` — configuration, constants, session paths, UI helpers, zones
- ``thesis.stage_1_data`` — OHLCV preparation
- ``thesis.stage_2_features`` — feature engineering
- ``thesis.stage_3_labels`` — triple-barrier labels
- ``thesis.stage_4_training`` — validation, GRU, LightGBM, walk-forward training
- ``thesis.stage_5_backtest`` — application-demo backtest
- ``thesis.stage_6_reporting`` — report and charts

Detailed API pages are intentionally not included until generated Sphinx API
stubs exist. This avoids broken references during documentation builds.

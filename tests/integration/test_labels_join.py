"""Integration tests for generate_labels join-path behavior.

NOTE: generate_labels / LabelsSchema removed in refactor.
Tests skipped pending rewrite for build_labels API.
"""

import pytest


@pytest.mark.integration
@pytest.mark.skip(reason="generate_labels / LabelsSchema removed in refactor")
class TestLabelsJoin:
    pass

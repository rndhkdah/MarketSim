from __future__ import annotations

import pytest

from marketsim.layer1.betas import derive_betas
from marketsim.layer1.checks import EARLY_CYCLE, RECESSION, assert_rotation


@pytest.mark.validation
def test_rotation_sets(cfg, io) -> None:
    betas = derive_betas(io, cfg)
    assert_rotation(io, cfg, betas)
    # Named tops — order inside the set is not a golden.
    assert set(EARLY_CYCLE) <= set(cfg.codes)
    assert set(RECESSION) <= set(cfg.codes)

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_committee.prereg import load_prereg

ROOT = Path(__file__).resolve().parents[1]


def test_committed_prereg_template_loads_and_matches_locked_decisions():
    p = load_prereg(ROOT / "prereg.yaml")
    assert p.metric == "accuracy" and p.alpha == 0.05
    assert len(p.price_tables) == 3
    assert all(t.assumed for t in p.price_tables) and p.error_cost.assumed
    assert p.decision_policy.breakeven_requires_ci_above_zero is True
    assert p.decision_policy.latency_in_decision is False


def test_load_prereg_rejects_invalid_yaml_content(tmp_path):
    bad = tmp_path / "prereg.yaml"
    bad.write_text("alpha: 2\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_prereg(bad)

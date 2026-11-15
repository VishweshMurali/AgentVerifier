import dataclasses
from pathlib import Path

from agent_committee import model_cache, quick, warm


def test_prepares_the_same_models_a_test_would_load(tmp_path, monkeypatch):
    """Pre-training saves under the quick test's own keys, then skips what is saved."""
    from agent_committee import scenarios
    from agent_committee.arms.splits import make_arms_split
    from agent_committee.data import load_dev

    monkeypatch.setattr(model_cache, "CACHE_DIR", tmp_path / "models")
    small = sorted(load_dev(Path("data")), key=lambda c: c.id)[:600]
    monkeypatch.setattr("agent_committee.data.load_dev", lambda _d: small)
    real = scenarios.get_scenario
    monkeypatch.setattr(scenarios, "get_scenario", lambda n, r: dataclasses.replace(real(n, r)))
    logs: list[str] = []
    warm.warm(Path("."), "banking77", log=logs.append)
    assert any("trained" in line for line in logs)
    split = make_arms_split(small)
    for name in ("naive_bayes", "linear_svm", "random_forest", "boosting"):
        key = quick.free_model_key(name, split)
        assert (tmp_path / "models" / f"{name}-{key}.pkl").is_file()
        _, _, loaded = quick.load_or_fit_free(Path("."), name, split, None)
        assert loaded  # the test path loads exactly what was prepared
    logs.clear()
    warm.warm(Path("."), "banking77", log=logs.append)
    assert logs and all("already saved" in line for line in logs)
    assert model_cache.read_summary(Path("."), "banking77") is not None


def test_ready_scenarios_include_banking77_and_built_ones():
    names = warm.ready_scenarios(Path("."))
    assert names[0] == "banking77" and "clinc150" in names

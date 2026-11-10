from collections import Counter

from agent_committee.arms.splits import ARMS_SPLIT_SEED, make_arms_split, normalize_text
from agent_committee.data import load_dev
from agent_committee.schemas import Case


def _cases(n_labels=4, per=50):
    return [
        Case(id=f"c{k}-{i:03d}", text=f"text {k} number {i}", label=f"lab{k}")
        for k in range(n_labels)
        for i in range(per)
    ]


def test_partition_is_disjoint_and_complete():
    dev = _cases()
    s = make_arms_split(dev)
    ids = [c.id for part in (s.pool, s.tune, s.eval) for c in part]
    assert sorted(ids) == sorted(c.id for c in dev)
    assert len(set(ids)) == len(ids)


def test_deterministic_and_input_order_independent():
    dev = _cases()
    a = make_arms_split(dev)
    b = make_arms_split(list(reversed(dev)))
    assert a == b
    assert ARMS_SPLIT_SEED == 20260102


def test_text_overlap_moves_to_pool():
    dev = _cases() + [Case(id=f"dup-{i}", text="  SAME   text ", label="lab0") for i in range(40)]
    s = make_arms_split(dev)
    pool_texts = {normalize_text(c.text) for c in s.pool}
    for c in s.tune + s.eval:
        assert normalize_text(c.text) not in pool_texts


def test_real_dev_counts_and_zero_overlap():
    dev = load_dev("data")
    s = make_arms_split(dev)
    counts = s.counts()
    print("arms split counts:", counts)
    assert sum(counts.values()) == len(dev)
    assert 7800 <= counts["pool"] <= 7900
    assert 450 <= counts["tune"] <= 500
    assert 1400 <= counts["eval"] <= 1500
    pool_texts = {normalize_text(c.text) for c in s.pool}
    assert not any(normalize_text(c.text) in pool_texts for c in s.tune + s.eval)
    assert set(Counter(c.label for c in s.eval)) == set(Counter(c.label for c in dev))


# ---------- grouped DEV (v2 P3b) ----------
import pytest  # noqa: E402


def _grouped(n_labels=3, groups_per_label=20, per_group=3):
    return [
        Case(id=f"l{lab}g{g}c{i}", text=f"label {lab} group {g} case {i}", label=f"l{lab}",
             group=f"l{lab}g{g}")
        for lab in range(n_labels)
        for g in range(groups_per_label)
        for i in range(per_group)
    ]  # fmt: skip


def test_grouped_split_keeps_every_group_in_one_part_and_is_deterministic():
    cases = _grouped()
    split = make_arms_split(cases)
    parts = [{c.group for c in part} for part in (split.pool, split.tune, split.eval)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert sum(split.counts().values()) == len(cases)
    assert {c.label for c in split.pool} == {c.label for c in cases}
    assert make_arms_split(list(reversed(cases))) == split


def test_grouped_split_moves_a_whole_group_when_a_text_repeats_in_the_pool():
    cases = _grouped()
    first = make_arms_split(cases)
    scored = first.eval[0]
    copy = Case(id="zz-copy", text=scored.text, label=scored.label, group="zz-new")
    split = make_arms_split([*cases, copy])
    pool_texts = {normalize_text(c.text) for c in split.pool}
    assert not any(normalize_text(c.text) in pool_texts for c in split.tune + split.eval)
    pool_groups = {c.group for c in split.pool}
    assert not {c.group for c in split.tune + split.eval} & pool_groups


def test_grouped_split_refuses_mixed_groups_and_too_few_groups():
    cases = _grouped()
    with pytest.raises(ValueError, match="some DEV cases carry a group"):
        make_arms_split([*cases, Case(id="x", text="x", label="l0")])
    with pytest.raises(ValueError):
        make_arms_split(_grouped(n_labels=2, groups_per_label=2, per_group=2))


def test_grouped_moves_repeat_until_no_scored_text_is_in_the_pool(monkeypatch):
    """A chain: the pool holds t1; eval group g1 holds t1 and t2; tune group g2 holds t2. One
    pass moves g1 (t1) and so brings t2 into the pool; only a second pass moves g2."""
    from agent_committee.arms import splits

    def case(i, text, group, label="a"):
        return Case(id=f"c{i}", text=text, label=label, group=group)

    pool = [case(0, "t1", "p0"), case(1, "own", "p1", "b")]
    g1 = [case(2, "t1", "g1"), case(3, "t2", "g1")]
    g2 = [case(4, "t2", "g2")]
    other = [case(5, "fresh tune", "g3"), case(6, "fresh eval", "g4", "b")]
    cuts = iter([(pool, [*g1, *g2, *other]), ([*g1, other[1]], [*g2, other[0]])])
    monkeypatch.setattr(splits, "_group_cut", lambda *_a: next(cuts))
    got = splits._grouped_split([*pool, *g1, *g2, *other], seed=0)
    assert {c.group for c in got.pool} == {"p0", "p1", "g1", "g2"}
    assert [c.text for c in got.tune] == ["fresh tune"] and [c.text for c in got.eval] == [
        "fresh eval"
    ]

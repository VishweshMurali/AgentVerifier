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

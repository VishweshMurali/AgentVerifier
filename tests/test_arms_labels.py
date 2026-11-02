from agent_committee.arms.labels import build_label_book, humanize, label_key
from agent_committee.arms.splits import make_arms_split
from agent_committee.data import load_dev
from agent_committee.schemas import Case


def test_humanize_odd_labels():
    assert humanize("Refund_not_showing_up") == "refund not showing up"
    assert humanize("reverted_card_payment?") == "reverted card payment"


def test_resolve_matches_odd_labels():
    pool = [
        Case(id=f"a{i}", text=f"refund missing {i}", label="Refund_not_showing_up")
        for i in range(5)
    ] + [
        Case(id=f"b{i}", text=f"payment reverted {i}", label="reverted_card_payment?")
        for i in range(5)
    ]
    book = build_label_book(pool)
    assert book.resolve("refund not showing up") == "Refund_not_showing_up"
    assert book.resolve("Reverted_Card_Payment") == "reverted_card_payment?"
    assert book.resolve("nope") is None
    assert label_key("A_b?") == "ab"


def test_book_is_deterministic_and_shaped():
    pool = make_arms_split(load_dev("data")).pool
    a = build_label_book(pool)
    b = build_label_book(list(reversed(pool)))
    assert a == b
    assert len(a.labels) == 77
    assert list(a.labels) == sorted(a.labels)
    for info in a.infos:
        assert len(info.terms) == 8
        assert len(info.examples) == 3
        assert all(e in {c.text for c in pool if c.label == info.label} for e in info.examples)
    assert "card" in " ".join(a.get("card_arrival").terms)

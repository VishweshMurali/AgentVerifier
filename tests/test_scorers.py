from agent_committee.prereg import load_prereg
from agent_committee.scorers import exact_label


def test_exact_label_needs_an_exact_match():
    assert exact_label("card_arrival", "card_arrival")
    assert not exact_label("Card_Arrival", "card_arrival")
    assert not exact_label(None, "card_arrival")


def test_prereg_names_this_scorer():
    assert load_prereg().scorer.target == "agent_committee.scorers:exact_label"

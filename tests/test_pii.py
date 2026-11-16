"""The PII masking layer for a reader's own data: what it masks and what it leaves alone."""

import pytest

from agent_committee import pii


@pytest.mark.parametrize(
    ("text", "want", "kind"),
    [
        ("mail me at jo.smith+x@mail.example.co.uk please", "mail me at [EMAIL] please", pii.EMAIL),
        ("card 4111 1111 1111 1111 was declined", "card [CARD] was declined", pii.CARD),
        ("card 4111-1111-1111-1111.", "card [CARD].", pii.CARD),
        ("IBAN GB82 WEST 1234 5698 7654 32 is mine", "IBAN [IBAN] is mine", pii.IBAN),
        ("my ssn is 078-05-1120", "my ssn is [ID_NUMBER]", pii.SSN),
        ("logged in from 192.168.10.4 today", "logged in from [IP_ADDRESS] today", pii.IP),
        ("call +44 20 7946 0958 now", "call [PHONE] now", pii.PHONE),
        ("call (555) 123-4567 now", "call [PHONE] now", pii.PHONE),
        ("I live at 12 Oak Street, Leeds", "I live at [ADDRESS], Leeds", pii.ADDRESS),
        ("account 123456789012 is locked", "account [LONG_NUMBER] is locked", pii.NUMBER),
    ],
)
def test_masks_each_kind(text, want, kind):
    masked, found = pii.mask(text)
    assert masked == want
    assert found == {kind: 1}


@pytest.mark.parametrize(
    "text",
    [
        "I was charged $1,250.00 twice on 2024-03-05",
        "the transfer of 2500 euros took 3 days",
        "my order 12345 arrived at 10:30",
        "version 1.2.3 crashed 4 times",
        "top up failed after 15 minutes, why?",
        "Where is my new card? I have been waiting a week!",
    ],
)
def test_leaves_ordinary_text_alone(text):
    masked, found = pii.mask(text)
    assert masked == text
    assert not found


def test_a_number_that_fails_the_card_check_is_still_masked_as_a_long_number():
    masked, found = pii.mask("ref 4111111111111112")
    assert masked == "ref [LONG_NUMBER]"
    assert found == {pii.NUMBER: 1}


def test_a_grouped_number_that_is_not_a_card_is_masked_too():
    masked, found = pii.mask("my number is 4111 1111 1111 1112 thanks")
    assert masked == "my number is [LONG_NUMBER] thanks"
    assert found == {pii.NUMBER: 1}


def test_luhn():
    assert pii.luhn("4111 1111 1111 1111")
    assert not pii.luhn("4111 1111 1111 1112")
    assert not pii.luhn("1234")


def test_several_details_in_one_text():
    masked, found = pii.mask("a@b.com or c@d.org, tel 555-123-4567")
    assert masked == "[EMAIL] or [EMAIL], tel [PHONE]"
    assert found == {pii.EMAIL: 2, pii.PHONE: 1}


def test_mask_all_counts_kinds_and_changed_texts():
    texts = ["hello", "write to a@b.com", "a@b.com and c@d.org"]
    out, totals, changed = pii.mask_all(texts)
    assert out == ["hello", "write to [EMAIL]", "[EMAIL] and [EMAIL]"]
    assert totals == {pii.EMAIL: 3}
    assert changed == 2


def test_masking_twice_changes_nothing():
    once, _ = pii.mask("a@b.com, +44 20 7946 0958, 4111 1111 1111 1111, 12 Oak Street")
    twice, found = pii.mask(once)
    assert twice == once and not found


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Hi Ana, my card is blocked", "Hi [NAME], my card is blocked"),
        ("my name is Ana Maria Diaz and I need help", "my name is [NAME] and I need help"),
        ("Mr Diaz called about a refund", "Mr [NAME] called about a refund"),
        ("Thanks, Ana", "Thanks, [NAME]"),
        ("see https://bank.example/x?id=5 or @ana_d", "see [URL] or [URL]"),
        ("born 12/03/1988 in Leeds", "born [DATE] in Leeds"),
        ("my date of birth is 3 March 1990", "my date of birth is [DATE]"),
        ("parcel to AB12 3CD", "parcel to [POSTCODE]"),
    ],
)
def test_masks_names_links_birth_dates_and_postcodes(text, want):
    assert pii.mask(text)[0] == want


@pytest.mark.parametrize(
    "text",
    [
        "I'm Sorry to bother you",
        "I am Unable to log in",
        "Dear Team, my transfer is pending",
        "this is Monday and I paid 5 pounds",
        "charged on 2024-03-05 twice",
    ],
)
def test_ordinary_words_and_transaction_dates_are_left_alone(text):
    assert pii.mask(text)[0] == text

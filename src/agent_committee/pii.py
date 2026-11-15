"""Personal details in a reader's own data are masked before anything is saved (owner request
2026-11-14). Plain code, standard library only, no model.

`mask(text)` replaces each detail with a tag such as [EMAIL] and counts what it replaced. It runs
on every uploaded message before the split is built, so the stored file, the trained models, the
audit bundle and anything sent to a hosted agent only ever hold the masked text.

What it finds: email addresses, payment card numbers (checked with the Luhn sum), IBANs, US
social security numbers, IP addresses, phone numbers, street addresses in the common English form
("12 Oak Street") and any other number of nine or more digits, also when written in groups
(account, customer or ID numbers).

What it cannot find: names of people and places, and details written out in words. The page says
so; this layer lowers the risk, it does not make data anonymous.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

EMAIL = "EMAIL"
CARD = "CARD"
IBAN = "IBAN"
SSN = "ID_NUMBER"
IP = "IP_ADDRESS"
PHONE = "PHONE"
ADDRESS = "ADDRESS"
NUMBER = "LONG_NUMBER"
KINDS = (EMAIL, CARD, IBAN, SSN, IP, PHONE, ADDRESS, NUMBER)

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_SSN = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")
_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_PHONE = re.compile(r"(?<![\w+])(?:\+\d|\(\d|\d)[\d\s().-]{6,18}\d(?!\w)")
_ADDRESS = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w'-]*\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
    r"Square|Sq|Terrace|Close|Crescent)\b\.?"
)
_NUMBER = re.compile(r"(?<![\w.])\d{9,}(?![\w.])")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def luhn(number: str) -> bool:
    """True when the digits pass the check sum payment cards use."""
    digits = [int(d) for d in _digits(number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _is_ip(text: str) -> bool:
    return all(int(part) <= 255 for part in text.split("."))


def _is_phone(text: str) -> bool:
    """9 to 15 digits, written the way phone numbers are (a +, brackets or separators), so a
    price or a plain count is left alone."""
    n = len(_digits(text))
    return 9 <= n <= 15 and (text.startswith(("+", "(")) or bool(re.search(r"[\s().-]", text)))


# checked in this order: the first pattern to claim a stretch of text keeps it
_RULES: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...] = (
    (EMAIL, _EMAIL, None),
    (IBAN, _IBAN, lambda t: len(t.replace(" ", "")) >= 15),
    (CARD, _CARD, luhn),
    (SSN, _SSN, None),
    (IP, _IP, _is_ip),
    (PHONE, _PHONE, _is_phone),
    (ADDRESS, _ADDRESS, None),
    (NUMBER, _NUMBER, None),
)


def mask(text: str) -> tuple[str, Counter[str]]:
    """The text with every found detail replaced by its [TAG], and how many of each kind."""
    found: Counter[str] = Counter()
    for kind, pattern, accept in _RULES:

        def swap(match: re.Match[str], kind: str = kind, accept=accept) -> str:
            if accept is not None and not accept(match.group()):
                if kind != CARD:
                    return match.group()
                kind = NUMBER  # card-shaped but not a card: still a long account or ID number
            found[kind] += 1
            return f"[{kind}]"

        text = pattern.sub(swap, text)
    return text, found


def mask_all(texts: list[str]) -> tuple[list[str], dict[str, int], int]:
    """Every text masked; the count per kind (only kinds found); how many texts were changed."""
    out, totals, changed = [], Counter(), 0
    for text in texts:
        masked, found = mask(text)
        out.append(masked)
        totals.update(found)
        changed += masked != text
    return out, {k: totals[k] for k in KINDS if totals[k]}, changed

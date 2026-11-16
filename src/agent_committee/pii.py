"""Personal details in a reader's own data are masked before anything is saved (owner request
2026-11-14). Plain code, standard library only, no model.

`mask(text)` replaces each detail with a tag such as [EMAIL] and counts what it replaced. It runs
on every uploaded message before the split is built, so the stored file, the trained models, the
audit bundle and anything sent to a hosted agent only ever hold the masked text.

What it finds: email addresses, web links and @handles, payment card numbers (checked with the
Luhn sum), IBANs, US social security numbers, IP addresses, phone numbers, street addresses in the
common English form ("12 Oak Street"), UK postcodes, full dates (such as a birth date), any other
number of nine or more digits (also in groups: account, customer or ID numbers), and the names of
people when something gives them away: "my name is Ana Diaz", "I'm Ana", "Hi Ana,", "Mr Diaz",
"Thanks, Ana". It errs toward masking too much.

What it cannot find: a name with no such cue, names of places, and details written out in words.
The page says so; this layer lowers the risk, it does not make data anonymous.
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
URL = "URL"
DATE = "DATE"
POSTCODE = "POSTCODE"
NAME = "NAME"
KINDS = (EMAIL, URL, CARD, IBAN, SSN, IP, PHONE, ADDRESS, POSTCODE, DATE, NAME, NUMBER)

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
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+|(?<![\w@])@[A-Za-z_][\w.]{2,}")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b")
_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"  # noqa: E501
_DATE_VALUE = (
    rf"(?:\d{{1,2}}[/.-]\d{{1,2}}[/.-](?:19|20)?\d{{2}}|(?:19|20)\d{{2}}-\d{{2}}-\d{{2}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+(?:19|20)\d{{2}}|{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}})"
)
# only a BIRTH date is personal; a transaction date ("charged on 2024-03-05") is left alone
_DOB = re.compile(
    rf"(?P<cue>(?i:born|dob|d\.o\.b\.?|date of birth|birth date|birthday)(?:\s+on|\s+is|:)?\s*)(?P<d>{_DATE_VALUE})"  # noqa: E501
)
# names: only where something gives them away, so ordinary capitalised words are left alone
_PERSON = r"[A-Z][a-z'-]{1,}(?:\s+[A-Z][a-z'-]{1,}){0,2}"
_NAME = re.compile(
    rf"(?P<cue>(?i:my name is|my name's|name is|name:|i am|i'm|this is|called|"
    rf"(?:mr|mrs|ms|miss|dr|prof|sir|madam)\.?|hi|hello|hey|dear|thanks,|thank you,|regards,|"
    rf"best,|cheers,|sincerely,)\s+)(?P<who>{_PERSON})"
)
_NOT_NAMES = {
    "Sorry",
    "There",
    "Please",
    "Thanks",
    "Thank",
    "Hello",
    "Hi",
    "Hey",
    "Team",
    "Support",
    "Customer",
    "Help",
    "Again",
    "Not",
    "Just",
    "Still",
    "Very",
    "Unable",
    "Having",
    "Trying",
    "Looking",
    "Writing",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Calling",
    "Contacting",
    "Locked",
    "Charged",
    "Stuck",
    "Concerned",
    "Worried",
    "Happy",
    "Angry",
}


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


def _swap_name(match: re.Match[str]) -> str:
    first = match.group("who").split()[0]
    return match.group() if first in _NOT_NAMES else f"{match.group('cue')}[{NAME}]"


# checked in this order: the first pattern to claim a stretch of text keeps it
_RULES: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...] = (
    (EMAIL, _EMAIL, None),
    (URL, _URL, None),
    (IBAN, _IBAN, lambda t: len(t.replace(" ", "")) >= 15),
    (CARD, _CARD, luhn),
    (SSN, _SSN, None),
    (IP, _IP, _is_ip),
    (PHONE, _PHONE, _is_phone),
    (ADDRESS, _ADDRESS, None),
    (POSTCODE, _POSTCODE, None),
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

    # names last: a cue followed by a capitalised word; a tag such as [EMAIL] is never one
    def named(match: re.Match[str]) -> str:
        out = _swap_name(match)
        if out != match.group():
            found[NAME] += 1
        return out

    text = _NAME.sub(named, text)

    def born(match: re.Match[str]) -> str:
        found[DATE] += 1
        return f"{match.group('cue')}[{DATE}]"

    text = _DOB.sub(born, text)
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

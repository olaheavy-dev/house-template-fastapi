"""What counts as personal data to remove: the project's list of detectors (#78, decision 0006).

This file is yours, like prices.py: edit, add or remove detectors, and a template update leaves it
alone. Each is a RegexDetector (pattern, an optional check such as a checksum, an optional word that
must appear within 30 characters before) or any object with `kind` and `find(text)`. The defaults
are a narrow set of contact, bank and ID formats, tuned to find what they can without touching
property text (prices, dates, postcodes, bands). Names and addresses are not guessed: the app passes
the exact values as `known` (app/llm/core/redaction.py).

These are not complete and not anonymisation. Every pattern must run in linear time: no nested
quantifiers; the possessive quantifier `++` and a `(?<![...])` lookbehind keep a match from being
retried at every position of a long run (Python 3.11 and later).
"""

from __future__ import annotations

from app.llm.core.redaction import Detector, RegexDetector, context, luhn_valid, nhs_valid

__all__ = ["DETECTORS"]


def _uk_phone(matched: str) -> bool:
    digits = "".join(c for c in matched if c.isdigit())
    if matched.startswith("+"):  # +44 7700 900123 or +44 (0)20 ...: drop 44 and any (0)
        digits = digits[2:]
        if matched.replace(" ", "").replace("-", "").startswith("+44(0)"):
            digits = digits[1:]
        digits = "0" + digits
    return len(digits) == 11 and digits[1] in "1237"


def _card_valid(matched: str) -> bool:
    digits = [c for c in matched if c.isdigit()]
    return len(set(digits)) > 1 and luhn_valid(matched)


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]{0,6}\.?"

DETECTORS: tuple[Detector, ...] = (
    RegexDetector(
        "EMAIL",
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9-]++(?:\.[A-Za-z0-9-]++)++",
    ),
    # UK numbers starting 01, 02, 03 or 07: 0 or +44 (with an optional (0)), then ten digits that
    # may be spaced, hyphenated or bracketed. The first digit after the prefix is checked below.
    RegexDetector(
        "PHONE",
        r"(?<![\w+])(?<!\d[ -])(?:\+44[ -]?(?:\(0\)[ -]?)?|\(?0)\d(?:\)?[ -]?\d){9}(?![ -]?\d)",
        check=_uk_phone,
    ),
    # Two letters, six digits, a final letter A to D: upper case, or lower case straight after the
    # digits (a lone "a" after a space is a word, as in "LN123456 a freehold title"). The prefix
    # letters are not validated, so this errs towards removing. A number written without its last
    # letter is not found.
    RegexDetector(
        "NINO",
        r"(?<![A-Za-z0-9])[A-Za-z]{2}(?:[ \u00a0]?\d{2}){3}"
        r"(?:[ \u00a0]?[A-D]|(?<=\d)[a-d])(?![A-Za-z0-9])",
    ),
    # GB IBAN layout; the mod-97 checksum is not required.
    RegexDetector(
        "IBAN",
        r"(?<![A-Za-z0-9])(?i:GB)\d{2}[ ]?[A-Za-z]{4}(?:[ ]?\d{4}){3}[ ]?\d{2}(?![A-Za-z0-9])",
    ),
    # The digit lookarounds keep a match from starting or ending inside a longer run of digits that
    # has spaces or hyphens in it (an energy certificate reference, a grouped amount).
    RegexDetector(
        "SORT_CODE",
        r"(?<!\d)(?<!\d[ -])\d{2}[- ]?\d{2}[- ]?\d{2}(?![ -]?\d)",
        before=context(r"sort[ -]?code"),
    ),
    RegexDetector(
        "ACCOUNT",
        r"(?<!\d)(?<!\d[ -])\d{4}[ ]?\d{4}(?![ -]?\d)",
        before=context(r"\bacc(?:ount)?\b"),
    ),
    # 13 to 19 digits in one run, or the usual groupings (4-4-4-4, optionally a last group of up to
    # three digits, and Amex 4-6-5), passing Luhn. Other groupings (3-3-3-3-3, 4-4-4-4-4) are not
    # cards. A number of one repeated digit is refused: 19 zeros pass Luhn.
    RegexDetector(
        "CARD",
        r"(?<!\d)(?<!\d[ -])"
        r"(?:\d{13,19}|\d{4}(?:[ -]\d{4}){3}(?:[ -]\d{1,3})?|\d{4}[ -]\d{6}[ -]\d{5})"
        r"(?![ -]?\d)",
        check=_card_valid,
    ),
    RegexDetector(
        "NHS",
        r"(?<!\d)\d{3}[ -]?\d{3}[ -]?\d{4}(?!\d)",
        check=nhs_valid,
        before=context(r"\bNHS\b"),
    ),
    RegexDetector(
        "DOB",
        r"(?<!\d)(?:\d{1,2}([/.-])\d{1,2}\1(?:\d{4}|\d{2})"
        r"|\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}(?:st|nd|rd|th)?[ ]+(?i:" + _MONTH + r")[ ]+\d{4})(?!\d)",
        before=context(r"date\s+of\s+birth|\bdob\b|\bborn\b"),
    ),
)

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


def _card_prefix_ok(digits: str) -> bool:
    """Whether the number starts like a card issuer's range: Visa 4; Mastercard 51-55 and
    2221-2720; American Express 34 and 37; Discover 6011, 644-649 and 65; Diners Club 30, 36, 38
    and 39; JCB 3528-3589; UnionPay 62; Maestro 5018, 5020, 5038, 5893, 6304, 6759 and 6761-6763
    (the UK ones among them). Taken from the Wikipedia summary of the issuers' ranges
    (https://en.wikipedia.org/wiki/Payment_card_number, read 9 Nov 2026); not checked against the
    schemes' own documents. A meter number or order number passes Luhn about 1 time in 10 and
    starts like a card about half of those, so this halves the false positives; it does not remove
    them. A number in a range not listed here is not found."""
    two, three, four = int(digits[:2]), int(digits[:3]), int(digits[:4])
    return (
        digits[0] == "4"
        or 51 <= two <= 55
        or 2221 <= four <= 2720
        or two in (34, 37, 30, 36, 38, 39, 62, 65)
        or four == 6011
        or 644 <= three <= 649
        or 3528 <= four <= 3589
        or four in (5018, 5020, 5038, 5893, 6304, 6759)
        or 6761 <= four <= 6763
    )


def _card_valid(matched: str) -> bool:
    digits = "".join(c for c in matched if c.isdigit())
    return len(set(digits)) > 1 and luhn_valid(digits) and _card_prefix_ok(digits)


# A UK phone number as it is usually written (not with arbitrary spaces): 07700900123, 07700 900123,
# 07700 900 123, 0113 496 0000, 020 7946 0321, and the +44 forms.
_STD_PHONE = (
    r"(?:0\d{10}|0\d{4} \d{6}|0\d{4} \d{3} \d{3}|0\d{3} \d{3} \d{4}|0\d{2} \d{4} \d{4}"
    r"|\+44 ?\d{4} ?\d{6}|\+44 ?\d{2} ?\d{4} ?\d{4}|\+44 ?\d{3} ?\d{3} ?\d{4})"
)
_PHONE_CORE = r"(?:\+44[ -]?(?:\(0\)[ -]?)?|\(?0)\d(?:\)?[ -]?\d){9}"
# After a phone number: no more digits, not even after a hyphen; and after a space no digit unless
# what follows is a number of one to three digits (a count, a house number), or the start of
# another standard phone number (so a row of phone numbers separated by spaces is found).
_PHONE_AHEAD = r"(?!-?\d)(?! (?!\d{1,3}(?![ -]?\d))(?!" + _STD_PHONE + r")\d)"


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]{0,6}\.?"

DETECTORS: tuple[Detector, ...] = (
    RegexDetector(
        "EMAIL",
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9-]++(?:\.[A-Za-z0-9-]++)++",
    ),
    # UK numbers starting 01, 02, 03 or 07: 0 or +44 (with an optional (0)), then ten digits that
    # may be spaced, hyphenated or bracketed. The first digit after the prefix is checked below.
    # Not straight after a digit and a space or hyphen (a longer run of digits, such as a reference
    # or an amount), except that after a digit and a space a number written in a standard UK
    # grouping still counts ("Flat 2 07700 900123"); and not before more digits (see
    # _PHONE_AHEAD).
    RegexDetector(
        "PHONE",
        r"(?<![\w+])(?<!\d[ -])"
        + _PHONE_CORE
        + _PHONE_AHEAD
        + r"|(?<=\d )"
        + _STD_PHONE
        + _PHONE_AHEAD,
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
        before=context(r"sort[\s_-]*codes?"),
    ),
    RegexDetector(
        "ACCOUNT",
        r"(?<!\d)(?<!\d[ -])\d{4}[ ]?\d{4}(?![ -]?\d)",
        before=context(r"\bacc(?:ount)?(?:s|[\s_-]*(?:no|num\w*))?\b|\bacct\b"),
    ),
    # 13 to 19 digits in one run, or the usual groupings (4-4-4-4, Amex 4-6-5), passing Luhn and
    # starting like an issuer's range. Other groupings (3-3-3-3-3, 4-4-4-4-4) are not cards. A
    # number of one repeated digit is refused. Two rules of one kind:
    # - the number alone, which may be followed by an expiry date or a security code (a space and a
    #   digit group is refused only when it is a fourth-and-later group of four, or the start of a
    #   longer grouped number), so "4111 1111 1111 1111 12/27" and "... 123" are found, and so is
    #   a 16-digit reference followed by a date (it errs towards removing);
    # - the 4-4-4-4 grouping with a last group of one to three digits (a 17 to 19 digit card).
    RegexDetector(
        "CARD",
        r"(?<!\d)(?<!\d[ -])"
        r"(?:\d{13,19}|\d{4}(?:[ -]\d{4}){3}|\d{4}[ -]\d{6}[ -]\d{5})"
        r"(?!-?\d)(?! \d{4})(?! \d{1,3}[ -]\d)",
        check=_card_valid,
    ),
    RegexDetector(
        "CARD",
        r"(?<!\d)(?<!\d[ -])\d{4}(?:[ -]\d{4}){3}[ -]\d{1,3}(?![ -]?\d)",
        check=_card_valid,
    ),
    RegexDetector(
        "NHS",
        r"(?<!\d)\d{3}[ -]?\d{3}[ -]?\d{4}(?!\d)",
        check=nhs_valid,
        before=context(r"\bNHS\b|\bnhs[\s_-]*(?:no|num\w*)\b"),
    ),
    RegexDetector(
        "DOB",
        r"(?<!\d)(?:\d{1,2}([/.-])\d{1,2}\1(?:\d{4}|\d{2})"
        r"|\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}(?:st|nd|rd|th)?[ ]+(?i:" + _MONTH + r")[ ]+\d{4})(?!\d)",
        before=context(
            r"date\s+of\s+birth|\bdob\b|\bd[.\s]o[.\s]b\b|\bborn\b|birth[\s_-]*(?:date|day)"
        ),
    ),
)

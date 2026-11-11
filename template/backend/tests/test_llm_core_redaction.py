"""Personal-data redaction in the framework-neutral core (#78): no database, no framework, no model.

Every expected value is worked out by hand from the rules in docs/decisions/0006 and the numbers
below, not by running the code:

- Phones are from Ofcom's drama ranges (07700 900xxx, 020 7946 0xxx), which are never allocated.
- IBAN GB82 WEST 1234 5698 7654 32 is the standard example and passes mod 97.
- Card 4111 1111 1111 1111 passes Luhn: from the right, the odd positions add 1+1+1+1+1+1+1+1 = 8
  and the even positions, each doubled (4 -> 8, 1 -> 2 and so on), add 8+2+2+2+2+2+2+2 = 22, so the
  total is 8 + 22 = 30 = 0 mod 10. Changing the last digit to 2 makes the total 31, which fails.
- NHS 943 476 5919 passes mod 11: weights 10 down to 2 give 90+36+24+28+42+30+20+27+2 = 299;
  299 = 27 * 11 + 2, so the check digit is 11 - 2 = 9, which is the last digit. Ending in 8 fails.

The tests use the default detectors in app/llm/redaction_rules.py, the project's own list.
"""

from __future__ import annotations

import ast
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.llm.core.redaction import (
    Redacted,
    RedactionFailed,
    Redactor,
    Scope,
    Span,
    StreamRestorer,
    choose_detectors,
    current_scope,
    redaction_scope,
    restore,
)
from app.llm.redaction_rules import DETECTORS
from app.problems import install_problem_handlers

REDACTOR = Redactor(DETECTORS)


def one(text: str) -> Redacted:
    return REDACTOR.redact([text])


# --- 1 to 3: contact details ---------------------------------------------------------------


def test_an_email_is_replaced_by_a_placeholder() -> None:
    result = one("Email me at priya.shah@example.com please")
    assert result.texts == ("Email me at [EMAIL_1] please",)
    assert result.mapping == {"[EMAIL_1]": "priya.shah@example.com"}
    assert result.counts == {"EMAIL": 1}


def test_the_same_value_gets_the_same_placeholder_and_a_new_one_the_next_number() -> None:
    result = one("a@example.com, then a@example.com again, then b@example.com")
    assert result.texts == ("[EMAIL_1], then [EMAIL_1] again, then [EMAIL_2]",)
    assert result.mapping == {"[EMAIL_1]": "a@example.com", "[EMAIL_2]": "b@example.com"}
    assert result.counts == {"EMAIL": 3}  # occurrences replaced, not distinct values


def test_uk_phone_numbers_are_replaced_in_the_common_formats() -> None:
    result = one("Call 07700 900123, office 020 7946 0321 or +44 7700 900123.")
    assert result.texts == ("Call [PHONE_1], office [PHONE_2] or [PHONE_3].",)
    assert result.counts == {"PHONE": 3}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call (020) 7946 0321.", "Call [PHONE_1]."),
        ("Call +44 (0)20 7946 0321.", "Call [PHONE_1]."),
        ("Call 07700900123.", "Call [PHONE_1]."),
        # Too short, or not a UK number range: left alone.
        ("Call 07700 90012.", "Call 07700 90012."),
        ("Call 0800 123 4567.", "Call 0800 123 4567."),
    ],
)
def test_phone_formats_and_near_misses(text: str, expected: str) -> None:
    assert one(text).texts == (expected,)


# --- 4 to 9: ID and bank numbers -----------------------------------------------------------


def test_a_national_insurance_number_is_replaced_with_or_without_spaces() -> None:
    result = one("NI: AB 12 34 56 C and QQ123456C.")
    assert result.texts == ("NI: [NINO_1] and [NINO_2].",)
    assert result.counts == {"NINO": 2}


def test_a_reference_that_is_not_a_national_insurance_number_is_left_alone() -> None:
    # "AB123456" has no final letter. A known miss in the other direction: a NINO written without
    # its last letter is not recognised, and a prefix is never checked, so QQ123456C is replaced.
    assert one("Ref AB123456").texts == ("Ref AB123456",)


def test_an_iban_is_replaced_whether_or_not_its_checksum_is_right() -> None:
    # GB82 passes mod 97; GB00 does not. The checksum is not required, so both go.
    result = one("Pay GB82 WEST 1234 5698 7654 32 or GB00 WEST 1234 5698 7654 32.")
    assert result.texts == ("Pay [IBAN_1] or [IBAN_2].",)


def test_a_sort_code_and_account_number_need_their_words_nearby() -> None:
    result = one("Sort code 12-34-56, account 12345678.")
    assert result.texts == ("Sort code [SORT_CODE_1], account [ACCOUNT_1].",)
    assert result.counts == {"SORT_CODE": 1, "ACCOUNT": 1}
    # A date and an eight-digit reference with no such words are not bank details.
    text = "Move in 01-03-27, ref 12345678"
    assert one(text).texts == (text,)
    assert one("acc no 87654321").texts == ("acc no [ACCOUNT_1]",)


def test_sort_codes_and_account_numbers_may_be_spaced() -> None:
    assert one("Sort code 12 34 56").texts == ("Sort code [SORT_CODE_1]",)
    assert one("sort code 123456").texts == ("sort code [SORT_CODE_1]",)
    assert one("Sort code 12 34 56, account 1234 5678.").texts == (
        "Sort code [SORT_CODE_1], account [ACCOUNT_1].",
    )
    # Not at the edge of a longer run of digits, even with spaces in it.
    for text in ("account 1234 5678 9012", "sort code 12 34 56 78"):
        assert one(text).texts == (text,), text


def test_a_national_insurance_number_may_end_in_a_lower_case_letter_only_without_a_space() -> None:
    assert one("ab123456c").texts == ("[NINO_1]",)
    assert one("ab 12 34 56 C").texts == ("[NINO_1]",)
    text = "Title number LN123456 a freehold title"
    assert one(text).texts == (text,)  # also in the must-not-change test above


def test_a_card_number_needs_a_right_luhn_check_digit() -> None:
    assert one("Card 4111 1111 1111 1111 expires").texts == ("Card [CARD_1] expires",)
    assert one("Card 4111-1111-1111-1111 expires").texts == ("Card [CARD_1] expires",)
    wrong = "Card 4111 1111 1111 1112 expires"
    assert one(wrong).texts == (wrong,)


def _grouped(number: str) -> str:
    if len(number) == 15:  # American Express: 4-6-5
        return f"{number[:4]} {number[4:10]} {number[10:]}"
    return " ".join(number[i : i + 4] for i in range(0, len(number), 4))


# Test numbers that pass Luhn (checked by hand-running the algorithm on each; the Luhn check of
# 4111 1111 1111 1111 is worked out at the top of this file).
CARDS = {
    "Visa": "4111111111111111",
    "Mastercard": "5555555555554444",
    "Mastercard 2-series": "2223003122003222",
    "American Express": "378282246310005",
    "Discover": "6011111111111117",
    "Maestro": "6759649826438453",
    "Maestro 5018": "5018123412341207",
}


@pytest.mark.parametrize("brand", list(CARDS))
@pytest.mark.parametrize(
    "after", [" 12/27", " 123", " exp 12/27", ", 12/27 cvv 123", " 1 payment", ""]
)
def test_a_card_is_found_in_every_layout_and_with_an_expiry_date_or_security_code(
    brand: str, after: str
) -> None:
    number = CARDS[brand]
    for written in (number, _grouped(number), _grouped(number).replace(" ", "-")):
        assert one(f"Card {written}{after}").texts == (f"Card [CARD_1]{after}",), (brand, written)


def test_a_number_that_does_not_start_like_a_card_is_not_one() -> None:
    # 1234567890128 passes Luhn: from the right the digits 8,2,1,0,9,8,7,6,5,4,3,2,1 give
    # 8, 2*2=4, 1, 0*2=0, 9, 8*2=16->7, 7, 6*2=12->3, 5, 4*2=8, 3, 2*2=4, 1 = 60, a multiple of 10.
    # It starts 12, which no card issuer uses, so a meter number like it is left alone.
    text = "MPAN 1234567890128"
    assert one(text).texts == (text,)


def test_a_sixteen_digit_reference_that_passes_luhn_and_is_followed_by_a_date_is_removed() -> None:
    # The cost of finding "4111 1111 1111 1111 12/27": it errs towards removing.
    assert one("Ref 4111111111111111 12/03/2027").texts == ("Ref [CARD_1] 12/03/2027",)


def test_an_nhs_number_needs_its_word_and_a_right_check_digit() -> None:
    assert one("NHS number 943 476 5919").texts == ("NHS number [NHS_1]",)
    assert one("My nhs no. 9434765919").texts == ("My nhs no. [NHS_1]",)
    for text in ("Ref 943 476 5919", "NHS number 943 476 5918"):  # no word; wrong check digit
        assert one(text).texts == (text,)


def test_an_nhs_number_whose_check_digit_would_be_ten_is_never_valid() -> None:
    # 000 000 0060: the first nine digits are 0,0,0,0,0,0,0,0,6, weighted 10,9,...,2, so the sum is
    # 6 * 2 = 12. 12 = 1 * 11 + 1, so the check is 11 - 1 = 10, and a check of 10 means the number
    # is invalid (no NHS number is issued with it). The last digit is 0, so a check that reads 10 as
    # 0 would wrongly accept it.
    text = "NHS number 000 000 0060"
    assert one(text).texts == (text,)


def test_a_date_of_birth_needs_its_words_nearby() -> None:
    assert one("Date of birth: 03/02/1990").texts == ("Date of birth: [DOB_1]",)
    assert one("DOB 3.2.90").texts == ("DOB [DOB_1]",)
    assert one("Born 3 February 1990").texts == ("Born [DOB_1]",)
    text = "Move in 03/02/1990"
    assert one(text).texts == (text,)


# --- 10: what must not change --------------------------------------------------------------


def test_ordinary_property_text_is_left_exactly_as_it_was() -> None:
    text = (
        "Flat 3, 12 Example Road, Leeds LS1 4AP. Rent £1,250 pcm from 01/03/2027. EPC band C. "
        "Council tax band B. Deposit £1,440. Agent: Sam Patel, Example Lettings."
    )
    result = one(text)
    assert result.texts == (text,)
    assert result.mapping == {}
    assert result.counts == {}


def test_reference_numbers_and_words_that_look_like_identifiers_are_left_alone() -> None:
    # An energy certificate reference is five groups of four digits; its first sixteen digits must
    # not be tested as a card, nor any later group as a phone or account number.
    for text in (
        "Energy certificate 1648-7059-6402-3592-5634 valid",
        "EPC RRN 0120-2834-7690-2399-4415 dated 01/03/2027",
        "Rent £968 157 655 500 575 in total",  # a grouped amount, fifteen digits
        "Total 1 234 567 890 123 456 over the term",
        "Order number 0000000000000000000",  # one repeated digit passes Luhn, but is not a card
        # Seventeen fours: 9 undoubled digits give 36 and 8 doubled ones 8 each, so 36 + 64 = 100
        # passes Luhn, and 4 is Visa's prefix, so only the repeated-digit rule leaves it alone.
        "Ref 44444444444444444",
        "Title number LN123456 a freehold title",  # a lone "a" is a word, not a NINO suffix
        "Unit AB 12 34 56 a and b",
    ):
        assert one(text).texts == (text,), text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A house or flat number before a phone number is ordinary text.
        ("Flat 2 07700 900123", "Flat 2 [PHONE_1]"),
        ("Unit 5 020 7946 0321", "Unit 5 [PHONE_1]"),
        ("Call 2 07700 900123 now", "Call 2 [PHONE_1] now"),
        ("Flat 10 +44 7700 900123", "Flat 10 [PHONE_1]"),
        ("12 Example Road 0113 496 0000", "12 Example Road [PHONE_1]"),
        # A count or a house number after one, and two phones separated only by a space.
        ("Call 07700 900123 3 times", "Call [PHONE_1] 3 times"),
        ("Ring 07700 900123 12 Example Road", "Ring [PHONE_1] 12 Example Road"),
        ("020 7946 0321 5", "[PHONE_1] 5"),
        ("07700 900123 07700 900456", "[PHONE_1] [PHONE_2]"),
        # Any number of them in a row.
        (
            "07700 900123 07700 900456 07700 900789",
            "[PHONE_1] [PHONE_2] [PHONE_3]",
        ),
        # Not after a digit and a hyphen: that is one longer number.
        ("Flat 2-07700 900123", "Flat 2-07700 900123"),
        # Number-like runs that are not phones stay (the 21-digit meter number written in groups).
        ("S 10 123 456 12 3456 7890 123", "S 10 123 456 12 3456 7890 123"),
    ],
)
def test_a_phone_number_after_or_before_other_numbers(text: str, expected: str) -> None:
    assert one(text).texts == (expected,)


def test_no_generated_reference_or_grouped_amount_is_changed() -> None:
    rng = random.Random(78)  # noqa: S311

    def digits(count: int) -> str:
        return "".join(rng.choice("0123456789") for _ in range(count))

    for _ in range(3000):
        epc = "-".join(digits(4) for _ in range(5))
        amount = f"£{rng.choice('123456789')}{digits(rng.randint(0, 2))}" + "".join(
            f" {digits(3)}" for _ in range(rng.randint(1, 6))
        )
        for text in (f"Energy certificate {epc} valid", f"Rent {amount} in total"):
            assert one(text).texts == (text,), text


# --- 11 to 13: what the app says (known and allow) -----------------------------------------


@pytest.mark.anyio
async def test_known_values_are_replaced_case_insensitively_and_restored_as_given() -> None:
    known = {"NAME": ["Priya Shah"], "ADDRESS": ["12 Example Road, Leeds"]}
    async with redaction_scope(known=known):
        result = one("Priya Shah lives at 12 Example Road,  Leeds. Hi priya shah. Priya alone.")
    assert result.texts == ("[NAME_1] lives at [ADDRESS_1]. Hi [NAME_1]. Priya alone.",)
    # Restored in the caller's form, not the form found in the text.
    assert result.mapping == {"[NAME_1]": "Priya Shah", "[ADDRESS_1]": "12 Example Road, Leeds"}
    assert result.counts == {"NAME": 2, "ADDRESS": 1}
    # "Priya" alone is not "Priya Shah": a known miss, the cost of exact matching.
    assert restore("Hello [NAME_1]", result.mapping) == ("Hello Priya Shah", 0)


@pytest.mark.anyio
async def test_an_allowed_value_is_not_replaced_but_others_still_are() -> None:
    async with redaction_scope(allow=["020 7946 0321", "lettings@example.com"]):
        result = one("Agent 020 7946 0321, lettings@example.com. Mobile 07700 900123.")
    assert result.texts == ("Agent 020 7946 0321, lettings@example.com. Mobile [PHONE_1].",)
    assert result.counts == {"PHONE": 1}


@pytest.mark.anyio
async def test_the_conflict_log_can_be_switched_off_for_trace_masking() -> None:
    # Trace masking redacts every run payload; it passes log_conflicts=False so one conflicting
    # known value is not logged for each of them. The redaction itself is the same.
    async with redaction_scope(
        known={"EMAIL": ["priya.shah@example.com"]}, allow=["PRIYA.SHAH@example.com"]
    ):
        with capture_logs() as logs:
            result = REDACTOR.redact(["Write to priya.shah@example.com"], log_conflicts=False)
    assert result.texts == ("Write to [EMAIL_1]",)
    assert [e for e in logs if e["event"] == "redaction_allow_conflict"] == []


@pytest.mark.anyio
async def test_a_known_value_beats_the_allow_list_and_the_conflict_is_logged() -> None:
    async with redaction_scope(
        known={"EMAIL": ["priya.shah@example.com"]}, allow=["PRIYA.SHAH@example.com"]
    ):
        with capture_logs() as logs:
            result = one("Write to priya.shah@example.com")
    assert result.texts == ("Write to [EMAIL_1]",)
    conflicts = [entry for entry in logs if entry["event"] == "redaction_allow_conflict"]
    assert len(conflicts) == 1
    assert "example.com" not in str(logs)  # the event never carries a value


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "known", "expected", "original"),
    [
        # A known value that covers only part of a detected value must not leave the rest behind:
        # the whole detected value is replaced (the union), and it keeps the detector's kind
        # because it is the longer member.
        (
            "Email priya.shah.1987@example.com",
            {"NAME": ["Priya"]},
            "Email [EMAIL_1]",
            "priya.shah.1987@example.com",
        ),
        (
            "Email priya.shah@example.com",
            {"NAME": ["Priya", "Shah"]},
            "Email [EMAIL_1]",
            "priya.shah@example.com",
        ),
        ("Ref AB 12 34 56 C", {"ADDRESS": ["12 34"]}, "Ref [NINO_1]", "AB 12 34 56 C"),
        # A known value that covers the whole detected value, and more, keeps its own kind.
        (
            "Flat 3, 07700 900123 is mine",
            {"ADDRESS": ["Flat 3, 07700 900123"]},
            "[ADDRESS_1] is mine",
            "Flat 3, 07700 900123",
        ),
        # The same extent: the known value wins the tie, and is restored as the app gave it.
        (
            "Mail PRIYA.SHAH@EXAMPLE.COM",
            {"ACCOUNT_EMAIL": ["priya.shah@example.com"]},
            "Mail [ACCOUNT_EMAIL_1]",
            "priya.shah@example.com",
        ),
    ],
)
async def test_a_known_value_that_overlaps_a_detected_one_replaces_their_union(
    text: str, known: dict[str, list[str]], expected: str, original: str
) -> None:
    async with redaction_scope(known=known):
        result = one(text)
    assert result.texts == (expected,)
    assert list(result.mapping.values()) == [original]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "allow", "expected"),
    [
        # An allowed value that covers only part of a detected value does not keep it.
        ("Email priya@leeds.ac.uk", ["Leeds"], "Email [EMAIL_1]"),
        ("Reach me at priya.example@gmail.com", ["Example"], "Reach me at [EMAIL_1]"),
        ("Call 07700 900123", ["900123"], "Call [PHONE_1]"),
        # One that contains the whole detected value keeps it.
        ("Email priya@leeds.ac.uk", ["priya@leeds.ac.uk"], "Email priya@leeds.ac.uk"),
        ("Call 07700 900123", ["Call 07700 900123"], "Call 07700 900123"),
    ],
)
async def test_an_allowed_value_keeps_a_detected_one_only_if_it_contains_all_of_it(
    text: str, allow: list[str], expected: str
) -> None:
    async with redaction_scope(allow=allow):
        assert one(text).texts == (expected,)


@pytest.mark.anyio
async def test_the_scope_lasts_for_the_block_only() -> None:
    assert current_scope().known == {}
    async with redaction_scope(known={"NAME": ["Priya Shah"]}):
        assert current_scope().known == {"NAME": ["Priya Shah"]}
    assert current_scope().known == {}
    assert one("Priya Shah").texts == ("Priya Shah",)


# --- 14 to 17: putting the originals back --------------------------------------------------


def test_restoring_puts_the_originals_back_and_counts_placeholders_it_does_not_know() -> None:
    mapping = one("Email me at priya.shah@example.com").mapping
    text, unrestored = restore("Sure, I will write to [EMAIL_1]. Not [EMAIL_7].", mapping)
    assert text == "Sure, I will write to priya.shah@example.com. Not [EMAIL_7]."
    assert unrestored == 1


def test_restoring_is_one_pass_so_an_original_that_looks_like_a_placeholder_stays() -> None:
    assert restore("[NAME_1]", {"[NAME_1]": "[NAME_2]", "[NAME_2]": "x"}) == ("[NAME_2]", 0)


def test_mangled_placeholders_are_counted_not_restored() -> None:
    mapping = {"[EMAIL_1]": "a@example.com"}
    text = "[EMAIL 1] [email_1] [EMAIL_1 ] [ EMAIL_1] [EMAIL-1] [EMAIL_2] [EMAIL_1]"
    # Six placeholder-shaped strings are not in the mapping; the last is, and is restored.
    assert restore(text, mapping) == (
        "[EMAIL 1] [email_1] [EMAIL_1 ] [ EMAIL_1] [EMAIL-1] [EMAIL_2] a@example.com",
        6,
    )


def feed_all(restorer: StreamRestorer, deltas: Iterable[str]) -> list[str]:
    return [restorer.feed(delta) for delta in deltas]


def test_a_stream_restores_a_placeholder_split_across_chunks() -> None:
    restorer = StreamRestorer({"[EMAIL_1]": "priya@example.com"})
    out = feed_all(restorer, ["Reply to [EM", "AIL_1] to", "day"])
    assert out == ["Reply to ", "priya@example.com to", "day"]
    assert restorer.flush() == ""


def test_a_stream_lets_a_bracket_that_is_not_a_placeholder_through() -> None:
    restorer = StreamRestorer({"[EMAIL_1]": "priya@example.com"})
    assert feed_all(restorer, ["Band [C", " or D"]) == ["Band ", "[C or D"]


def test_a_stream_gives_back_held_text_when_it_ends() -> None:
    restorer = StreamRestorer({"[EMAIL_1]": "priya@example.com"})
    assert restorer.feed("Reply to [EMAIL_") == "Reply to "
    assert restorer.flush() == "[EMAIL_"
    assert restorer.flush() == ""


def test_a_stream_with_an_empty_mapping_still_holds_a_possible_placeholder_and_counts_it() -> None:
    # The count must match restore() whatever the chunking, so the rule does not depend on the
    # mapping: "[C" could still become a placeholder-shaped string, so it waits.
    restorer = StreamRestorer({})
    assert restorer.feed("Band [C") == "Band "
    assert restorer.flush() == "[C"


def test_a_stream_counts_the_placeholders_it_could_not_restore() -> None:
    restorer = StreamRestorer({"[EMAIL_1]": "a@example.com"})
    out = feed_all(restorer, ["[EMAIL_7] and [EM", "AIL_1]"])
    assert "".join(out) + restorer.flush() == "[EMAIL_7] and a@example.com"
    assert restorer.unrestored == 1


def test_a_stream_gives_the_same_text_and_count_as_restore_for_any_chunking() -> None:
    mapping = {"[EMAIL_1]": "a@example.com", "[PHONE_12]": "07700 900123", "[A_1]": "z"}
    pieces = [
        "[",
        "]",
        "E",
        "M",
        "A",
        "I",
        "L",
        "_",
        "1",
        "2",
        "x",
        " ",
        "-",
        "[EMAIL_1]",
        "e",
        "[PHONE_12]",
        "[A_1]",
        "[EMAIL_7]",
        "[[",
        "_1]",
        "[A_",
        "PHONE_1",
        "[LONGKIND_",
        "[AVERYLONGKINDNAMEBEYONDTHELONGESTMAPPING_12]",
        "[EMAIL 1]",
        "\n",
    ]
    rng = random.Random(9)  # noqa: S311
    for _ in range(6000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
        cuts = sorted(rng.sample(range(len(text) + 1), k=rng.randint(0, min(8, len(text) + 1))))
        chunks = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]
        restorer = StreamRestorer(mapping)
        streamed = "".join(restorer.feed(chunk) for chunk in chunks) + restorer.flush()
        expected, unrestored = restore(text, mapping)
        assert streamed == expected, (text, chunks)
        assert restorer.unrestored == unrestored, (text, chunks)


def test_a_placeholder_already_in_the_input_is_skipped_and_never_restored() -> None:
    result = one("Keep [EMAIL_1] as written. Write to a@example.com")
    assert result.texts == ("Keep [EMAIL_1] as written. Write to [EMAIL_2]",)
    assert result.mapping == {"[EMAIL_2]": "a@example.com"}
    # What the model echoes of the literal one is not ours to restore, and is counted.
    assert restore("[EMAIL_1] [EMAIL_2]", result.mapping) == ("[EMAIL_1] a@example.com", 1)


def test_a_placeholder_in_a_later_message_is_skipped_in_an_earlier_one() -> None:
    result = REDACTOR.redact(["a@example.com", "see [EMAIL_1]"])
    assert result.texts == ("[EMAIL_2]", "see [EMAIL_1]")


# --- 18: numbering across the whole message list -------------------------------------------


def test_numbers_stay_the_same_when_the_history_grows() -> None:
    history = ["Hi a@example.com", "Call 07700 900123"]
    first = REDACTOR.redact(history)
    assert first.texts == ("Hi [EMAIL_1]", "Call [PHONE_1]")
    longer = REDACTOR.redact([*history, "Reply", "b@example.com and a@example.com"])
    assert longer.texts[:2] == first.texts  # the prompt cache keeps matching
    assert longer.texts[2:] == ("Reply", "[EMAIL_2] and [EMAIL_1]")


def test_numbers_are_counted_per_kind() -> None:
    result = one("a@example.com 07700 900123 b@example.com 020 7946 0321")
    assert result.texts == ("[EMAIL_1] [PHONE_1] [EMAIL_2] [PHONE_2]",)


# --- 21: failing closed --------------------------------------------------------------------


class Exploding:
    kind = "EMAIL"

    def find(self, text: str) -> Iterable[Span]:
        raise ValueError(f"cannot read {text}")


def test_a_detector_that_raises_blocks_the_call_and_leaks_nothing() -> None:
    value = "priya.shah@example.com"
    with capture_logs() as logs, pytest.raises(RedactionFailed) as raised:
        Redactor([Exploding()]).redact([f"Write to {value}"])
    assert raised.value.status == 503
    assert value not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None  # the original error, which holds the text, is dropped
    assert logs == [
        {"event": "llm_redaction_failed", "error_type": "ValueError", "log_level": "error"}
    ]


def test_the_failure_is_a_503_problem_with_no_value_in_it() -> None:
    app = FastAPI()
    install_problem_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        with capture_logs():
            Redactor([Exploding()]).redact(["priya.shah@example.com"])

    response = TestClient(app).get("/boom")
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert "priya" not in response.text
    assert "nothing was sent" in response.json()["detail"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "known", "allow", "expected", "original"),
    [
        # The exact case from review: the allowed email contains the whole detected value, and the
        # known name overlaps it, so the union is replaced (a known value beats the allow-list).
        (
            "Write to priya.shah@example.com",
            {"NAME": ["Priya"]},
            ["priya.shah@example.com"],
            "Write to [EMAIL_1]",
            "priya.shah@example.com",
        ),
        # A known value inside the allowed email, in its domain.
        (
            "Email priya.shah.1987@leeds.ac.uk now",
            {"PLACE": ["Leeds"]},
            ["priya.shah.1987@leeds.ac.uk"],
            "Email [EMAIL_1] now",
            "priya.shah.1987@leeds.ac.uk",
        ),
        # A known value overlapping the edge of the allowed value and reaching past it.
        (
            "Mail priya@leeds.ac.uk today",
            {"NOTE": ["uk today"]},
            ["priya@leeds.ac.uk"],
            "Mail [EMAIL_1]",
            "priya@leeds.ac.uk today",
        ),
        # No known value touches it: the allow-list still keeps it.
        (
            "Mail priya@leeds.ac.uk, Sam Patel",
            {"NAME": ["Sam Patel"]},
            ["priya@leeds.ac.uk"],
            "Mail priya@leeds.ac.uk, [NAME_1]",
            "Sam Patel",
        ),
    ],
)
async def test_a_known_value_beats_the_allow_list_even_when_it_only_overlaps_a_detected_value(
    text: str,
    known: dict[str, list[str]],
    allow: list[str],
    expected: str,
    original: str,
) -> None:
    async with redaction_scope(known=known, allow=allow):
        result = one(text)
    assert result.texts == (expected,)
    assert list(result.mapping.values()) == [original]


@pytest.mark.anyio
async def test_a_known_value_and_a_detected_one_that_differ_never_share_a_placeholder() -> None:
    # straße@example.com is known; strasse@example.com is a different address (the case folding of
    # ß is not applied), found by the detector. Each is restored as itself.
    text = "straße@example.com strasse@example.com"
    async with redaction_scope(known={"EMAIL": ["straße@example.com"]}):
        result = one(text)
    assert result.texts == ("[EMAIL_1] [EMAIL_2]",)
    assert result.mapping == {
        "[EMAIL_1]": "straße@example.com",
        "[EMAIL_2]": "strasse@example.com",
    }
    assert restore(result.texts[0], result.mapping) == (text, 0)


@pytest.mark.anyio
async def test_a_known_value_s_normal_form_is_not_a_detected_value_s_key() -> None:
    # "Strasse" is known, with "\u00df": its normal form is "strasse". A detector of the same kind
    # finds the different text "strasse" (the known pattern does not fold \u00df to ss). The two
    # must not share a placeholder, or the second would be restored as the first.
    class Plain:
        kind = "NAME"

        def find(self, text: str) -> Iterable[Span]:
            at = text.find("strasse")
            return [Span(at, at + 7, "NAME")] if at != -1 else []

    async with redaction_scope(known={"NAME": ["Stra\u00dfe"]}):
        result = Redactor([Plain()]).redact(["Stra\u00dfe strasse"])
    assert result.texts == ("[NAME_1] [NAME_2]",)
    assert restore(result.texts[0], result.mapping)[0] == "Stra\u00dfe strasse"


@pytest.mark.anyio
async def test_known_values_that_normalise_equal_share_one_placeholder() -> None:
    known = {"NAME": ["Priya Shah", "priya  shah"]}
    async with redaction_scope(known=known):
        result = one("Priya Shah and PRIYA SHAH")
    assert result.texts == ("[NAME_1] and [NAME_1]",)
    assert len(result.mapping) == 1


@pytest.mark.anyio
async def test_a_value_that_starts_later_and_reaches_further_extends_the_union() -> None:
    # known "Flat 3, 07700" is 0-13 and the phone is 8-20: the phone reaches further, so the union
    # must run to 20 and not stop at 13 (which would leave " 900123" in the clear).
    async with redaction_scope(known={"ADDRESS": ["Flat 3, 07700"]}):
        result = one("Flat 3, 07700 900123 ok")
    assert result.texts == ("[ADDRESS_1] ok",)
    assert result.mapping == {"[ADDRESS_1]": "Flat 3, 07700 900123"}


@pytest.mark.anyio
async def test_a_chain_of_three_overlapping_values_is_one_union() -> None:
    # known 0-13, phone 8-20, known 14-23: the first and last do not overlap, only through the
    # phone. The longest member (the first known value, 13 characters) gives the kind.
    known = {"ADDRESS": ["Flat 3, 07700"], "NOTE": ["900123 ok"]}
    async with redaction_scope(known=known):
        result = one("Flat 3, 07700 900123 ok")
    assert result.texts == ("[ADDRESS_1]",)
    assert result.mapping == {"[ADDRESS_1]": "Flat 3, 07700 900123 ok"}
    assert result.counts == {"ADDRESS": 1}


@pytest.mark.anyio
async def test_a_known_value_is_found_in_either_unicode_normal_form() -> None:
    composed, decomposed = "Jos\u00e9 Silva", "Jose\u0301 Silva"
    for known, text in ((composed, decomposed), (decomposed, composed), (composed, composed)):
        async with redaction_scope(known={"NAME": [known]}, allow=[]):
            result = one(f"Dear {text}, hello")
        assert result.texts == ("Dear [NAME_1], hello",), (known, text)
        assert result.mapping == {"[NAME_1]": known}  # restored in the caller's form
    async with redaction_scope(allow=[decomposed]):
        assert one(f"Agent {composed} on 07700 900123").texts[0].startswith(f"Agent {composed} on")


@pytest.mark.anyio
async def test_a_bad_kind_or_a_single_string_never_puts_a_value_in_the_error() -> None:
    for known in (
        {"Priya Shah": "Priya Shah"},
        {"Priya Shah": ["Priya Shah"]},
        {"NAME": "Priya Shah"},
    ):
        with pytest.raises((TypeError, ValueError)) as raised:
            async with redaction_scope(known=known):
                pass  # pragma: no cover
        assert "Priya" not in str(raised.value), known


# --- what a caller or a detector may hand in ----------------------------------------------


class Fixed:
    def __init__(self, kind: str, spans: list[object]) -> None:
        self.kind = kind
        self._spans = spans

    def find(self, text: str) -> Iterable[Span]:
        return self._spans  # type: ignore[return-value]


@pytest.mark.parametrize("kind", ["name", "FULL-NAME", "e mail", "", "1NAME", "NAME]", "X" * 60])
def test_a_kind_that_would_not_be_restored_is_refused(kind: str) -> None:
    with pytest.raises(ValueError, match="kind"):
        Redactor([Fixed(kind, [])])
    with pytest.raises(ValueError, match="kind"):
        Scope(known={kind: ["Priya"]})


@pytest.mark.anyio
async def test_a_scope_refuses_a_bad_kind_and_a_detector_span_with_one_fails_closed() -> None:
    with pytest.raises(ValueError, match="kind"):
        async with redaction_scope(known={"name": ["Priya"]}):
            pass  # pragma: no cover
    with capture_logs(), pytest.raises(RedactionFailed):
        Redactor([Fixed("EMAIL", [Span(0, 3, "e mail")])]).redact(["hello world"])


@pytest.mark.parametrize(
    "span",
    [
        Span(-5, 3, "EMAIL"),
        Span(3, 1, "EMAIL"),
        Span(2, 99, "EMAIL"),
        Span(2, 2, "EMAIL"),
        (0, 3, "EMAIL"),
        Span(0.5, 3, "EMAIL"),  # type: ignore[arg-type]
    ],
)
def test_a_detector_span_outside_the_text_or_empty_fails_closed(span: object) -> None:
    with capture_logs() as logs, pytest.raises(RedactionFailed):
        Redactor([Fixed("EMAIL", [span])]).redact(["hello world"])
    assert [entry["event"] for entry in logs] == ["llm_redaction_failed"]
    assert "hello" not in str(logs)


def test_a_single_string_where_a_list_is_expected_is_refused() -> None:
    with pytest.raises(TypeError, match="texts"):
        REDACTOR.redact("priya.shah@example.com")
    with pytest.raises(TypeError, match="texts"):
        REDACTOR.redact(["fine", 3])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="known values"):
        Scope(known={"NAME": "Priya Shah"})
    with pytest.raises(TypeError, match="allow"):
        Scope(allow="7")


@pytest.mark.anyio
async def test_redaction_scope_refuses_a_single_string_too() -> None:
    with pytest.raises(TypeError, match="known values"):
        async with redaction_scope(known={"NAME": "Priya Shah"}):
            pass  # pragma: no cover
    with pytest.raises(TypeError, match="allow"):
        async with redaction_scope(allow="7"):
            pass  # pragma: no cover


def test_nothing_to_redact_gives_nothing_back() -> None:
    empty = REDACTOR.redact([])
    assert (empty.texts, empty.mapping, empty.counts) == ((), {}, {})
    blank = REDACTOR.redact([""])
    assert (blank.texts, blank.mapping, blank.counts) == (("",), {}, {})


def test_a_detected_value_has_its_own_placeholder_for_each_exact_spelling() -> None:
    # The original is put back exactly, so a different case is a different value: A@X.COM comes
    # back as A@X.COM, never as a@x.com. (Known values are the other way round: matched ignoring
    # case, restored as the app gave them.)
    result = one("a@example.com, A@EXAMPLE.COM, a@example.com")
    assert result.texts == ("[EMAIL_1], [EMAIL_2], [EMAIL_1]",)
    assert result.mapping == {"[EMAIL_1]": "a@example.com", "[EMAIL_2]": "A@EXAMPLE.COM"}


# --- more information passed in never leaks more -------------------------------------------

_WORDS = [
    *("Priya", "Shah", "Leeds", "office", "agency", "Flat", "2", "12", "Road", "call", "on", "at"),
    *("co", "uk", "ac", "x", "com", "sam", "acc", "account", "sort code", "NHS", "born"),
]
_TOKENS = ["07700 900123", "020 7946 0321", "+44 7700 900456", "0113 496 0000", "AB123456C",
           "4111 1111 1111 1111", "GB82 WEST 1234 5698 7654 32", "12-34-56", "12345678",
           "943 476 5919", "01/02/1990"]  # fmt: skip


def _random_token(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.15:
        local = rng.choice(["priya.shah", "sam", "office", "a.b"])
        return local + "@" + rng.choice(["leeds.ac.uk", "agency.co.uk", "x.com"])
    if roll < 0.30:
        return rng.choice(_TOKENS)
    if roll < 0.40:
        return str(rng.randint(0, 99999))
    return rng.choice(_WORDS)


def _spans(text: str, patterns_of: list[str]) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for value in patterns_of:
        words = value.split()
        if words:
            pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", re.I)
            found.extend(m.span() for m in pattern.finditer(text))
    return found


def _covered(text: str, out: str, mapping: dict[str, str]) -> list[bool] | None:
    """Which characters of `text` ended up inside a placeholder; None if the output cannot be
    lined up with the text (a known value matched in a different case or spacing)."""
    covered = [False] * len(text)
    names = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(map(re.escape, names))) if names else None
    at = pos = 0
    for match in pattern.finditer(out) if pattern else []:
        literal = out[pos : match.start()]
        if text[at : at + len(literal)] != literal:
            return None
        at += len(literal)
        original = mapping[match.group()]
        if text[at : at + len(original)] != original:
            return None
        for index in range(at, at + len(original)):
            covered[index] = True
        at += len(original)
        pos = match.end()
    return covered if text[at:] == out[pos:] else None


def property_leaks(cases: int, seed: int) -> tuple[int, int]:
    """Random texts with random known and allow values taken from the text (allow values often
    hold a whole detected value, known values often overlap one). Returns (leaks, checked): every
    known match must be replaced, and so must every detected value except one that an allowed
    value contains whole and that no known match touches."""
    rng = random.Random(seed)  # noqa: S311
    leaks = checked = 0
    for _ in range(cases):
        text = rng.choice([" ", ", ", "  "]).join(
            _random_token(rng) for _ in range(rng.randint(3, 12))
        )
        parts = text.split()

        def pick(parts: list[str] = parts) -> str:
            i = rng.randrange(len(parts))
            value = " ".join(parts[i : i + rng.randint(1, 3)])
            if rng.random() < 0.3 and "@" in value:
                value = rng.choice(re.split(r"[@.]", value)) or value
            return value

        known = {"NAME": [v for v in (pick() for _ in range(rng.randint(0, 2))) if v in text]}
        allow = [pick() for _ in range(rng.randint(0, 3))]
        with capture_logs():
            result = REDACTOR.redact([text], Scope(known=known, allow=allow))
        covered = _covered(text, result.texts[0], dict(result.mapping))
        if covered is None:
            continue
        known_spans = _spans(text, known["NAME"])
        allow_spans: list[tuple[int, int]] = []
        for span in sorted(_spans(text, allow)):  # allowed values that overlap act as one
            if allow_spans and span[0] < allow_spans[-1][1]:
                allow_spans[-1] = (allow_spans[-1][0], max(allow_spans[-1][1], span[1]))
            else:
                allow_spans.append(span)
        for start, end, _kind in (
            (s.start, s.end, s.kind) for d in DETECTORS for s in d.find(text)
        ):
            kept = any(a <= start and end <= b for a, b in allow_spans) and not any(
                start < ke and ks < end for ks, ke in known_spans
            )
            if kept:
                continue
            checked += 1
            leaks += not all(covered[start:end])
        for start, end in known_spans:
            checked += 1
            leaks += not all(covered[start:end])
    return leaks, checked


def test_passing_in_known_and_allowed_values_never_leaks_a_detected_or_known_value() -> None:
    leaks, checked = property_leaks(4000, seed=7)
    assert checked > 5000  # the cases did exercise the rules
    assert leaks == 0


# --- 27: linear time -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit",
    ["@", "1", "a", "0", " ", "1 ", "1-", "a@", "1.", "a.", "(", "GB82", "-", "9 4", "/", "0 "],
)
def test_the_default_patterns_run_in_linear_time(unit: str) -> None:
    # About 1 MiB of one repeated unit: a pattern that backtracks badly takes minutes here.
    text = unit * ((1 << 20) // len(unit))
    started = time.perf_counter()
    one(text)
    assert time.perf_counter() - started < 2.0, unit


def test_many_spans_of_two_lengths_run_in_linear_time() -> None:
    # 11-character phones, then 12-character ones, comma separated (a digit after a space ends a
    # match). Each later span is longer, which once made the overlap step insert at the front of
    # a list for every span: about 12 s at 4 MiB.
    size = 4 << 20
    text = "07700900123, " * (size // 2 // 13) + "07700 900123, " * (size // 2 // 14)
    started = time.perf_counter()
    result = one(text)
    assert time.perf_counter() - started < 4.0
    assert result.counts == {"PHONE": size // 2 // 13 + size // 2 // 14}


def test_known_values_run_in_linear_time_too() -> None:
    text = "a" + " " * (1 << 20)
    started = time.perf_counter()
    with capture_logs():
        REDACTOR.redact([text], scope=Scope(known={"NAME": ["a b"]}))
    assert time.perf_counter() - started < 2.0


# --- choosing detectors by kind (LLM_REDACT_KINDS) ------------------------------------------


def test_choose_detectors_keeps_only_the_named_kinds() -> None:
    chosen = choose_detectors(DETECTORS, " EMAIL , PHONE,")
    assert {d.kind for d in chosen} == {"EMAIL", "PHONE"}
    assert Redactor(chosen).redact(["a@example.com 07700 900123 QQ123456C"]).texts == (
        "[EMAIL_1] [PHONE_1] QQ123456C",
    )


def test_a_kind_with_two_detectors_keeps_both() -> None:
    assert len([d for d in choose_detectors(DETECTORS, "CARD") if d.kind == "CARD"]) == 2


def test_an_empty_list_chooses_no_detectors_so_only_known_values_are_replaced() -> None:
    assert choose_detectors(DETECTORS, "") == ()
    assert choose_detectors(DETECTORS, " , ") == ()


def test_an_unknown_kind_is_refused_naming_it_and_the_available_kinds() -> None:
    with pytest.raises(ValueError, match=r"names EMIAL, which no detector.*Available: .*EMAIL"):
        choose_detectors(DETECTORS, "EMIAL,PHONE")
    # Kinds are matched exactly: lower case is not a kind.
    with pytest.raises(ValueError, match="email"):
        choose_detectors(DETECTORS, "email")


# --- the neutral walkers and restorer import no framework (redaction_data.py) ---------------------


def test_the_neutral_redaction_walkers_import_only_stdlib_pydantic_and_the_core() -> None:
    from app.llm.core import redaction_data

    tree = ast.parse(Path(redaction_data.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            roots.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"pydantic", "app"}
    assert roots <= allowed, roots - allowed


def test_importing_the_neutral_redaction_walkers_loads_no_framework() -> None:
    # A fresh interpreter: the test process has LangChain loaded already.
    frameworks = (
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
        "langgraph",
        "langsmith",
        "openai",
        "anthropic",
        "agent_framework",
    )
    code = (
        "import sys, app.llm.core.redaction_data\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {frameworks!r}))"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


# --- redact_each: one result per text, numbered inside that text (#78) ----------------


def test_redact_each_numbers_inside_each_text_so_neighbours_do_not_matter() -> None:
    result = REDACTOR.redact_each(["a@example.com", "b@example.com a@example.com", "no data"])
    # redact() would number across the list (b@ would be [EMAIL_2]); here every text starts at 1.
    assert result.texts == ("[EMAIL_1]", "[EMAIL_1] [EMAIL_2]", "no data")
    assert result.counts == (1, 2, 0)
    assert result.kinds == {"EMAIL": 3}
    assert REDACTOR.redact(["a@example.com", "b@example.com"]).texts == ("[EMAIL_1]", "[EMAIL_2]")


def test_a_text_is_the_same_alone_or_with_neighbours_in_any_order() -> None:
    texts = ["call 07700 900123", "a@example.com and 07700 900123", "b@example.com"]
    together = REDACTOR.redact_each(texts).texts
    for index, text in enumerate(texts):
        assert REDACTOR.redact_each([text]).texts == (together[index],)
    assert REDACTOR.redact_each(texts[::-1]).texts == together[::-1]


@pytest.mark.anyio
async def test_redact_each_replaces_known_values_and_keeps_allowed_ones_in_every_text() -> None:
    async with redaction_scope(known={"NAME": ["Priya Shah"]}, allow=["020 7946 0321"]):
        result = REDACTOR.redact_each(
            ["Priya Shah rang 020 7946 0321", "Then priya shah and 07700 900123"]
        )
    assert result.texts == ("[NAME_1] rang 020 7946 0321", "Then [NAME_1] and [PHONE_1]")
    assert result.counts == (1, 2)
    assert result.kinds == {"NAME": 2, "PHONE": 1}


def test_a_placeholder_written_in_one_text_is_avoided_only_in_that_text() -> None:
    result = REDACTOR.redact_each(["[EMAIL_1] and a@example.com", "b@example.com"])
    assert result.texts == ("[EMAIL_1] and [EMAIL_2]", "[EMAIL_1]")


@pytest.mark.anyio
async def test_a_conflict_between_a_known_and_an_allowed_value_is_logged_once_for_the_list() -> (
    None
):
    async with redaction_scope(
        known={"EMAIL": ["priya.shah@example.com"]}, allow=["PRIYA.SHAH@example.com"]
    ):
        with capture_logs() as logs:
            result = REDACTOR.redact_each(["to priya.shah@example.com"] * 3)
    assert result.texts == ("to [EMAIL_1]",) * 3
    assert [e for e in logs if e["event"] == "redaction_allow_conflict"] == [
        {"event": "redaction_allow_conflict", "kind": "EMAIL", "log_level": "warning"}
    ]


class ExplodesOn:
    """Fails on a text holding `word`, finds nothing in any other."""

    kind = "EMAIL"

    def __init__(self, word: str) -> None:
        self._word = word

    def find(self, text: str) -> Iterable[Span]:
        if self._word in text:
            raise ValueError(f"cannot read {text}")
        return []


def test_one_text_that_fails_blocks_the_whole_list_and_leaks_nothing() -> None:
    with capture_logs() as logs, pytest.raises(RedactionFailed) as raised:
        Redactor([ExplodesOn("boom")]).redact_each(["fine", "boom priya.shah@example.com", "ok"])
    assert raised.value.status == 503
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert logs == [
        {"event": "llm_redaction_failed", "error_type": "ValueError", "log_level": "error"}
    ]


def test_redact_each_refuses_a_single_string_and_non_text() -> None:
    with pytest.raises(TypeError, match="texts"):
        REDACTOR.redact_each("priya.shah@example.com")
    with pytest.raises(TypeError, match="texts"):
        REDACTOR.redact_each(["fine", 3])  # type: ignore[list-item]


def test_redact_each_of_nothing_gives_nothing() -> None:
    result = REDACTOR.redact_each([])
    assert (result.texts, result.counts, result.kinds) == ((), (), {})


@pytest.mark.anyio
async def test_redact_each_builds_the_scopes_patterns_once_however_many_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.llm.core import redaction

    built: list[str] = []
    real = redaction._value_patterns

    def counting(value: str) -> list[re.Pattern[str]]:
        built.append(value)
        return real(value)

    monkeypatch.setattr(redaction, "_value_patterns", counting)
    async with redaction_scope(known={"NAME": ["Priya Shah"]}, allow=["020 7946 0321"]):
        REDACTOR.redact_each(["Priya Shah"] * 50)
    assert sorted(built) == ["020 7946 0321", "Priya Shah"]  # one value, one build, not 50


def test_redact_each_runs_in_linear_time_over_many_texts() -> None:
    texts = ["Mail a@example.com or call 07700 900123 today. "] * 20_000
    started = time.perf_counter()
    result = REDACTOR.redact_each(texts)
    assert time.perf_counter() - started < 4.0
    assert result.kinds == {"EMAIL": 20_000, "PHONE": 20_000}

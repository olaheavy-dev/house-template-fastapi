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

import time
from collections.abc import Iterable

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


def test_a_card_number_needs_a_right_luhn_check_digit() -> None:
    assert one("Card 4111 1111 1111 1111 expires").texts == ("Card [CARD_1] expires",)
    assert one("Card 4111-1111-1111-1111 expires").texts == ("Card [CARD_1] expires",)
    wrong = "Card 4111 1111 1111 1112 expires"
    assert one(wrong).texts == (wrong,)


def test_an_nhs_number_needs_its_word_and_a_right_check_digit() -> None:
    assert one("NHS number 943 476 5919").texts == ("NHS number [NHS_1]",)
    assert one("My nhs no. 9434765919").texts == ("My nhs no. [NHS_1]",)
    for text in ("Ref 943 476 5919", "NHS number 943 476 5918"):  # no word; wrong check digit
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


def test_a_stream_with_nothing_to_restore_holds_nothing_back() -> None:
    restorer = StreamRestorer({})
    assert restorer.feed("Band [C") == "Band [C"


def test_a_stream_counts_the_placeholders_it_could_not_restore() -> None:
    restorer = StreamRestorer({"[EMAIL_1]": "a@example.com"})
    out = feed_all(restorer, ["[EMAIL_7] and [EM", "AIL_1]"])
    assert "".join(out) + restorer.flush() == "[EMAIL_7] and a@example.com"
    assert restorer.unrestored == 1


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


def test_known_values_run_in_linear_time_too() -> None:
    text = "a" + " " * (1 << 20)
    started = time.perf_counter()
    with capture_logs():
        REDACTOR.redact([text], scope=Scope(known={"NAME": ["a b"]}))
    assert time.perf_counter() - started < 2.0

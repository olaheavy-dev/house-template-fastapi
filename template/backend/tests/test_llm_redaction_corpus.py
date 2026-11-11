"""The redaction corpus (tests/evals/redaction_corpus.py) against the default detectors, offline:
the same two properties the live leak eval checks on the wire, without a model. Every expected
count comes from how the corpus is built, not from running the detectors."""

from __future__ import annotations

import pytest

from app.llm.core.redaction import Redactor, Scope
from app.llm.redaction_rules import DETECTORS
from tests.evals.redaction_corpus import (
    KNOWN_MISSES,
    LEAK_CORPUS,
    MUST_NOT_CHANGE,
    Document,
    luhn_complete,
    nhs_complete,
)

REDACTOR = Redactor(DETECTORS)


def redact(document: Document) -> tuple[str, dict[str, int]]:
    result = REDACTOR.redact([document.text], Scope(known=document.known, allow=document.allow))
    return result.texts[0], dict(result.counts)


def test_the_corpus_has_twenty_documents_with_distinct_people_and_every_kind() -> None:
    assert len(LEAK_CORPUS) == 20
    assert len({d.name for d in LEAK_CORPUS}) == 20
    kinds = {kind for d in LEAK_CORPUS for kind in d.counts}
    assert kinds == {"NAME", "EMAIL", "PHONE", "NINO", "SORT_CODE", "ACCOUNT", "CARD", "NHS",
                     "DOB", "IBAN"}  # fmt: skip


@pytest.mark.parametrize("document", LEAK_CORPUS, ids=lambda d: d.name)
def test_no_seeded_value_survives_redaction_and_each_kind_is_counted_as_built(
    document: Document,
) -> None:
    text, counts = redact(document)
    for value in document.seeded:
        assert value not in text, value
    assert counts == document.counts


@pytest.mark.parametrize("document", LEAK_CORPUS, ids=lambda d: d.name)
def test_the_property_address_stays_while_the_users_details_go(document: Document) -> None:
    text, _ = redact(document)
    assert (
        "Flat 3, 12 Example Road, Leeds LS1 4AP" in text
    )  # allowed, and not a personal value here


@pytest.mark.parametrize("text", MUST_NOT_CHANGE)
def test_property_text_that_looks_like_personal_data_is_left_alone(text: str) -> None:
    assert REDACTOR.redact([text], Scope()).texts == (text,)


def test_the_known_misses_are_really_missed_so_the_eval_reports_them_honestly() -> None:
    # Decision 0006 and #97: names that are not told to the redactor, free-form addresses and a
    # NINO without its last letter are not found. If a detector starts finding one, this fails and
    # the list (and the documentation of the limit) should be updated.
    for label, text in KNOWN_MISSES:
        assert REDACTOR.redact([text], Scope()).texts == (text,), label


def test_the_check_digit_builders_follow_the_published_rules() -> None:
    # Luhn: 4111 1111 1111 1111 passes (the worked example in the core tests), so completing its
    # first 15 digits gives it back.
    assert luhn_complete("411111111111111") == "4111111111111111"
    # NHS 943 476 591: weighted sum 299 = 27 x 11 + 2, so the check digit is 11 - 2 = 9. Changing
    # the last digit from 1 to 6 adds 5 x 2 = 10 to the sum (309 = 28 x 11 + 1): the check digit
    # would be 10, which no number has.
    assert nhs_complete("943476591") == "9434765919"
    assert nhs_complete("943476596") is None

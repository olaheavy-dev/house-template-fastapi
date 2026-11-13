"""The context report's numbers and the trim policy (#46, app/llm/core/context.py), framework-free.

Every expected figure is worked out by hand below, from the rules in the module notes, not from
the code.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from app.llm.core.bounds import IMAGE_TOKENS, INPUT_MARGIN_TOKENS
from app.llm.core.context import (
    Entry,
    OutputTooLong,
    check_output_limit,
    count,
    input_limit,
    plan_trim,
    report,
    size_bound,
    usage_from_record,
    usage_record,
)
from app.llm.core.prices import Price, Rates, Window
from app.llm.core.usage import Usage
from app.llm.prices import EMBEDDING_PRICES, PRICES
from tests.test_llm_prices import latest_plausible_checked

CHECKED, SOURCE = "2026-11-13", "https://models.example/page"
RATES = Rates(1, 1, 1, 1, 1)


def window(tokens: int = 10_000, max_output: int = 1_000) -> Window:
    return Window(tokens, max_output, checked=CHECKED, source=SOURCE)


def price(win: Window | None = None, max_prompt_tokens: int | None = None) -> Price:
    return Price(
        RATES, checked=CHECKED, source=SOURCE, window=win, max_prompt_tokens=max_prompt_tokens
    )


def used(input_tokens: int, output_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


# --- the window entry -------------------------------------------------------------------------


def test_every_chat_model_records_its_window_with_date_and_source() -> None:
    latest = latest_plausible_checked(datetime.now(UTC))
    for model, entry in PRICES.items():
        assert entry.window is not None, f"{model} has no context window"
        assert date.fromisoformat(entry.window.checked) <= latest, model
        assert entry.window.source.startswith("https://"), model


def test_embedding_prices_need_no_window() -> None:
    assert all(entry.window is None for entry in EMBEDDING_PRICES.values())


@pytest.mark.parametrize(
    ("tokens", "max_output", "complaint"),
    [
        (0, 1, "tokens"),
        (10, 0, "max_output"),
        (True, 1, "tokens"),
        (10, 10, "smaller"),
        (10, 11, "smaller"),
    ],
)
def test_a_window_that_cant_be_right_is_refused(
    tokens: int, max_output: int, complaint: str
) -> None:
    with pytest.raises(ValueError, match=complaint):
        Window(tokens, max_output, checked=CHECKED, source=SOURCE)


def test_a_window_without_a_date_or_page_is_refused() -> None:
    with pytest.raises(ValueError, match="checked"):
        Window(10, 1, checked="13 Nov 2026", source=SOURCE)
    with pytest.raises(ValueError, match="model page"):
        Window(10, 1, checked=CHECKED, source="http://models.example/page")


# --- the limit (100%) -------------------------------------------------------------------------


def test_the_limit_is_the_window_less_the_room_kept_for_the_answer() -> None:
    # 10,000 - 1,000 = 9,000.
    assert input_limit(price(window()), max_tokens=1_000) == 9_000


def test_the_listed_models_limits() -> None:
    # 1,000,000 - 1,024 = 998,976 and 1,050,000 - 1,024 = 1,048,976. OpenAI's 272,000 is the
    # budget's separate line (prompt_limit), not part of this one.
    assert input_limit(PRICES["anthropic:claude-sonnet-5-5"], 1_024) == 998_976
    assert input_limit(PRICES["openai:gpt-6-luna"], 1_024) == 1_048_976


def test_no_window_means_no_limit() -> None:
    assert input_limit(price(), max_tokens=1_000) is None


def test_an_output_limit_above_the_models_is_refused_at_start_up() -> None:
    check_output_limit(price(window(max_output=1_000)), 1_000)
    check_output_limit(price(), 10**9)  # no window: nothing to check against
    with pytest.raises(OutputTooLong, match="1001"):
        check_output_limit(price(window(max_output=1_000)), 1_001)


# --- the count --------------------------------------------------------------------------------


def test_the_count_starts_from_the_last_calls_usage() -> None:
    # The last call read 1,000 and wrote 500 (the answer, now history); the 25-token message
    # stored after it is estimated: 1,000 + 500 + 25 = 1,525.
    entries = [
        Entry(role="user", size=40),
        Entry(role="assistant", size=30, usage=used(1_000, 500)),
        Entry(role="user", size=25),
    ]
    assert count(entries).tokens == 1_525
    assert count(entries).measured
    # The budget's bound is every message by its bytes, plus the margin: 95 + 1,024.
    assert count(entries).bound == 95 + INPUT_MARGIN_TOKENS


def test_the_latest_usage_wins() -> None:
    entries = [
        Entry(role="user", size=40),
        Entry(role="assistant", size=30, usage=used(1_000, 500)),
        Entry(role="user", size=25),
        Entry(role="assistant", size=60, usage=used(3_000, 200)),
    ]
    assert count(entries).tokens == 3_200


def test_without_usage_everything_is_estimated_with_the_margin() -> None:
    entries = [Entry(role="user", size=40), Entry(role="assistant", size=30)]
    # 40 + 30 + the margin for message framing.
    assert count(entries).tokens == 70 + INPUT_MARGIN_TOKENS
    assert not count(entries).measured
    assert count([]).tokens == INPUT_MARGIN_TOKENS


def test_the_agents_own_part_is_estimated_only_without_usage() -> None:
    # 500 for the system prompt and tools: in the estimate (40 + 30 + 500 + 1,024 = 1,594), and
    # in the bound. With usage, the call's input already held them.
    entries = [Entry(role="user", size=40), Entry(role="assistant", size=30)]
    assert count(entries, fixed=500).tokens == 1_594
    measured = [*entries[:1], Entry(role="assistant", size=30, usage=used(1_000, 500))]
    got = count(measured, fixed=500)
    assert (got.tokens, got.bound) == (1_500, 1_594)


def test_size_is_one_token_per_byte_plus_each_image() -> None:
    # "héllo" as JSON is 8 bytes: two quotes, four one-byte letters, and é, which takes two.
    assert size_bound("héllo") == 8
    image = {"type": "image", "url": "u"}
    assert size_bound([image]) == len('[{"type": "image", "url": "u"}]') + IMAGE_TOKENS


def test_a_usage_record_round_trips_and_a_bad_one_is_ignored() -> None:
    assert usage_from_record(usage_record(used(12, 3))) == used(12, 3)
    for bad in (
        None,
        "x",
        {"input_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": True, "output_tokens": 1},
    ):
        assert usage_from_record(bad) is None


# --- the report -------------------------------------------------------------------------------


def test_the_report_gives_the_share_of_the_limit_to_one_decimal() -> None:
    entries = [Entry(role="user", size=25), Entry(role="assistant", size=5, usage=used(1_000, 500))]
    got = report(entries, price(window()), max_tokens=1_000)
    # 1,500 of 9,000 is 16.666...%, so 16.7. The bound is 30 + 1,024; there is no prompt cap.
    assert got.as_dict() == {
        "tokens": 1_500,
        "bound": 1_054,
        "window": 10_000,
        "limit": 9_000,
        "prompt_limit": None,
        "percent": 16.7,
        "trimmed": 0,
        "measured": True,
    }


def test_with_a_prompt_cap_the_budgets_line_can_be_the_nearer_one() -> None:
    # 1,500 tokens of 9,000 is 16.7%, but the budget compares its bound, 2,000 + 1,000 + 1,024 =
    # 4,024, with the 5,000 cap: 80.48%, so 80.5 is the share shown.
    entries = [
        Entry(role="user", size=2_000),
        Entry(role="assistant", size=1_000, usage=used(1_000, 500)),
    ]
    got = report(entries, price(window(), max_prompt_tokens=5_000), max_tokens=1_000)
    assert (got.tokens, got.bound, got.limit, got.prompt_limit) == (1_500, 4_024, 9_000, 5_000)
    assert got.percent == 80.5


def test_a_cap_without_a_window_still_gives_a_share() -> None:
    entries = [Entry(role="user", size=976)]
    # 976 + 1,024 = 2,000 of a 4,000 cap: 50%.
    assert report(entries, price(max_prompt_tokens=4_000), max_tokens=1_000).percent == 50.0


def test_the_share_passes_100_when_a_conversation_outgrows_the_limit() -> None:
    entries = [Entry(role="assistant", size=5, usage=used(9_000, 450))]
    # 9,450 of 9,000 is 105%.
    assert report(entries, price(window()), max_tokens=1_000).percent == 105.0


def test_without_a_window_the_report_still_counts() -> None:
    entries = [Entry(role="assistant", size=5, usage=used(800, 200))]
    got = report(entries, price(), max_tokens=1_000)
    assert (got.tokens, got.window, got.limit, got.percent) == (1_000, None, None, None)
    assert got.prompt_limit is None


# --- the trim policy --------------------------------------------------------------------------

# A conversation of three turns. Index: role, size (tokens), and what it is.
#   0 system     100
#   1 user        50  the first message: always kept
#   2 assistant   30  asks for a tool  } one group: 430
#   3 tool       400  its result       }
#   4 assistant   60
#   5 user        40
#   6 assistant   20  asks for a tool  } one group: 320
#   7 tool       300  its result       }
#   8 assistant   70
#   9 user        45  the latest turn starts here: always kept (9 to 12: 360)
#  10 assistant   80  asks for a tool
#  11 tool       200
#  12 assistant   35
# Total 1,430. Never dropped: 100 + 50 + 360 = 510.
SIZES = [100, 50, 30, 400, 60, 40, 20, 300, 70, 45, 80, 200, 35]
ROLES = [
    "system",
    "user",
    "assistant",
    "tool",
    "assistant",
    "user",
    "assistant",
    "tool",
    "assistant",
    "user",
    "assistant",
    "tool",
    "assistant",
]


# The tool call ids that tie each call to its result.
LINKS = {
    i: frozenset({call})
    for call, pair in (("a", (2, 3)), ("b", (6, 7)), ("c", (10, 11)))
    for i in pair
}


def conversation(waiting: frozenset[int] = frozenset()) -> list[Entry]:
    return [
        Entry(
            role=role,  # type: ignore[arg-type]
            size=size,
            links=LINKS.get(index, frozenset()),
            waiting=index in waiting,
        )
        for index, (role, size) in enumerate(zip(ROLES, SIZES, strict=True))
    ]


def test_the_example_adds_up() -> None:
    assert sum(SIZES) == 1_430


def test_nothing_is_dropped_at_or_under_the_target() -> None:
    plan = plan_trim(conversation(), target=1_430)
    assert (set(plan.drop), plan.tokens, plan.fits) == (set(), 1_430, True)


def test_the_oldest_tool_call_goes_first_with_its_result() -> None:
    # 1,430 - 430 = 1,000: at the target, so it stops there.
    plan = plan_trim(conversation(), target=1_000)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({2, 3}, 1_000, True)


def test_old_tool_calls_go_before_any_turn() -> None:
    # 1,430 - 430 - 320 = 680, under 700. The latest turn's tool call (10, 11) stays.
    plan = plan_trim(conversation(), target=700)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({2, 3, 6, 7}, 680, True)


def test_then_whole_old_turns_go_oldest_first_but_never_the_first_message() -> None:
    # After both tool calls: 680, over 600. The first turn (1 to 4) loses what is left of it but
    # message 1: 680 - 60 = 620, still over. The second turn (5 to 8) goes whole: 620 - 40 - 70
    # = 510.
    plan = plan_trim(conversation(), target=600)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({2, 3, 4, 5, 6, 7, 8}, 510, True)


def test_when_even_that_is_not_enough_it_says_so() -> None:
    # Everything the rules allow goes and 510 is left, over 300.
    plan = plan_trim(conversation(), target=300)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({2, 3, 4, 5, 6, 7, 8}, 510, False)


def test_a_waiting_tool_call_is_kept_with_its_result() -> None:
    # 6 and 7 stay: 1,430 - 430 = 1,000; then message 4: 940; then 5 and 8: 830, over 700.
    plan = plan_trim(conversation(waiting=frozenset({6})), target=700)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({2, 3, 4, 5, 8}, 830, False)


def test_what_cant_be_trimmed_counts_towards_the_target() -> None:
    # 500 fixed + 1,430 = 1,930; dropping 2 and 3 leaves 1,500, the target.
    plan = plan_trim(conversation(), target=1_500, fixed=500)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({2, 3}, 1_500, True)


# A real MAF 1.25.0 run with an approval: the answer to the approval request is stored as a user
# message between the tool call and its result.
#   0 user       50  the first message
#   1 assistant 460  calls c1 and asks for approval a1  } one group, by their ids: 10,711
#   2 user       40  approves a1                        }
#   3 tool   10,211  c1's result                        }
#   4 assistant  30
#   5 user       20  the latest turn
#   6 assistant  25
# Total 10,836.
APPROVAL = [
    Entry(role="user", size=50),
    Entry(role="assistant", size=460, links=frozenset({"c1", "approval:a1"})),
    Entry(role="user", size=40, links=frozenset({"c1", "approval:a1"})),
    Entry(role="tool", size=10_211, links=frozenset({"c1"})),
    Entry(role="assistant", size=30),
    Entry(role="user", size=20),
    Entry(role="assistant", size=25),
]


def test_an_approved_call_goes_with_its_approval_and_its_result() -> None:
    # 10,836 - 10,711 = 125: the call, the approval and the result go as one.
    plan = plan_trim(APPROVAL, target=10_736)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({1, 2, 3}, 125, True)


def test_an_approval_is_not_a_turn_so_the_latest_turn_keeps_its_call() -> None:
    # The same call in the latest turn: 0 user, 1 answer, 2 user (the latest turn), then the call,
    # the approval, the result and the answer. Only message 1 may go: 10,836 - 30 = 10,806.
    latest = [APPROVAL[0], APPROVAL[4], APPROVAL[5], *APPROVAL[1:4], APPROVAL[6]]
    plan = plan_trim(latest, target=100)
    assert (set(plan.drop), plan.tokens, plan.fits) == ({1}, 10_806, False)


@pytest.mark.parametrize("entries", [conversation(), APPROVAL], ids=["turns", "approval"])
def test_nothing_that_shares_a_tool_call_is_ever_split(entries: list[Entry]) -> None:
    for target in range(0, 11_000, 10):
        drop = plan_trim(entries, target=target).drop
        for i, one in enumerate(entries):
            for j, other in enumerate(entries):
                if one.links & other.links:
                    assert (i in drop) == (j in drop), (target, i, j)

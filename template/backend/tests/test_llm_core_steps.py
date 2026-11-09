"""The step-limit rule (#34), with no framework and no database: call k of a run offers tools
while k < N; call N is made without tools and is the final answer.

Expected values are worked out by hand from that rule: with N = 3, call 1 and call 2 offer tools
and call 3 does not; a run has made `calls_made` calls when it asks for call `calls_made + 1`.
"""

from __future__ import annotations

import dataclasses

import pytest
from structlog.testing import capture_logs

from app.llm.core import steps


def test_the_event_and_flag_name_is_step_limit_reached() -> None:
    assert steps.STEP_LIMIT_REACHED == "step_limit_reached"


def test_the_default_is_13_model_calls() -> None:
    # 25 LangGraph steps (the old default) allowed ceil(25 / 2) = 13 model calls.
    assert steps.DEFAULT_MAX_MODEL_CALLS == 13


def test_with_three_calls_the_first_two_offer_tools_and_the_third_is_final() -> None:
    limit = steps.StepLimit(3)
    # calls_made is how many calls the run has already made when it asks for the next one.
    assert [limit.offers_tools(made) for made in (0, 1, 2)] == [True, True, False]
    assert [limit.is_final(made) for made in (0, 1, 2)] == [False, False, True]


def test_with_thirteen_calls_only_the_thirteenth_is_final() -> None:
    limit = steps.StepLimit(13)
    # Calls 1..12 (calls_made 0..11) offer tools; call 13 (calls_made 12) is the final answer.
    assert [limit.is_final(made) for made in range(13)] == [False] * 12 + [True]
    assert [limit.offers_tools(made) for made in range(13)] == [True] * 12 + [False]


def test_with_one_call_the_only_call_is_final() -> None:
    limit = steps.StepLimit(1)
    assert limit.is_final(0) is True
    assert limit.offers_tools(0) is False


@pytest.mark.parametrize("bad", [0, -1, -13])
def test_a_limit_below_one_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        steps.StepLimit(bad)


def test_a_limit_cannot_be_changed() -> None:
    limit = steps.StepLimit(3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        limit.max_model_calls = 4  # type: ignore[misc]


def test_reaching_the_limit_logs_one_warning_with_the_event_name() -> None:
    with capture_logs() as logs:
        steps.log_reached(steps.StepLimit(3))
    assert logs == [{"event": "step_limit_reached", "log_level": "warning", "max_model_calls": 3}]

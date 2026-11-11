"""How an exhausted budget reaches the client: problem responses. (How a model failure reaches the
client, never with the provider's message, is in test_llm_parity_budget.py.)"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.llm.core.budget import BudgetExhausted

pytestmark = pytest.mark.anyio


def test_the_budget_says_to_retry_at_midnight_utc() -> None:
    refused = BudgetExhausted(now=datetime(2026, 1, 2, 23, 59, 30, tzinfo=UTC))
    assert (refused.status, refused.headers["Retry-After"]) == (503, "30")

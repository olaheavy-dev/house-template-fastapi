"""How model failures and an exhausted budget reach the client: problem responses that never carry
the provider's message."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.llm.core.budget import BudgetExhausted

pytestmark = pytest.mark.anyio


def test_the_budget_says_to_retry_at_midnight_utc() -> None:
    refused = BudgetExhausted(now=datetime(2026, 1, 2, 23, 59, 30, tzinfo=UTC))
    assert (refused.status, refused.headers["Retry-After"]) == (503, "30")

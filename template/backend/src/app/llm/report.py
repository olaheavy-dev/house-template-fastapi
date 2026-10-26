"""Model spend from the llm_call table: calls, cost, tokens and latency, grouped as asked.

    make llm-report                      # the last 7 days, by day and model
    make llm-report days=30 by=feature   # any of: day, model, prompt, feature

It reads DATABASE_URL like the app (the local database by default). Run it against production
with that database's URL, or from the API image: `docker compose run --rm api python -m
app.llm.report`.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import ColumnElement, Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import create_engine, create_sessionmaker
from app.llm.budget import LlmCall, expire_stale, utc_today
from app.llm.prices import MICRO_USD
from app.settings import get_settings

GROUPS = {
    "day": LlmCall.day,
    "model": LlmCall.model,
    "prompt": LlmCall.prompt,
    "feature": LlmCall.feature,
}


@dataclass(frozen=True, slots=True)
class SpendRow:
    group: tuple[str, ...]
    calls: int
    failed: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    p50_latency_ms: int | None
    p95_latency_ms: int | None


async def spend(session: AsyncSession, *, since: date, by: Sequence[str]) -> list[SpendRow]:
    columns = [GROUPS[name] for name in by]
    latency = LlmCall.latency_ms

    def percentile(share: float) -> ColumnElement[float]:
        return func.percentile_cont(share).within_group(latency)

    stmt = (
        select(
            *columns,
            func.count(),
            func.count().filter(LlmCall.status == "failed"),
            func.coalesce(func.sum(LlmCall.cost_micro_usd), 0),
            func.coalesce(func.sum(LlmCall.input_tokens), 0),
            func.coalesce(func.sum(LlmCall.output_tokens), 0),
            func.coalesce(func.sum(LlmCall.reasoning_tokens), 0),
            func.cast(percentile(0.5), Integer),
            func.cast(percentile(0.95), Integer),
        )
        .where(LlmCall.day >= since, LlmCall.status != "reserved")
        .group_by(*columns)
        .order_by(*columns)
    )
    rows = []
    for row in await session.execute(stmt):
        values = tuple(row)
        group = tuple("none" if v is None else str(v) for v in values[: len(columns)])
        calls, failed, cost, given, out, reasoning, p50, p95 = values[len(columns) :]
        # Postgres sums bigints as numeric (a Decimal here): back to whole numbers.
        rows.append(
            SpendRow(
                group,
                calls,
                failed,
                int(cost) / MICRO_USD,
                int(given),
                int(out),
                int(reasoning),
                p50,
                p95,
            )
        )
    return rows


def render(rows: Sequence[SpendRow], by: Sequence[str]) -> str:
    header = [*by, "calls", "failed", "cost USD", "in", "out", "reasoning", "p50 ms", "p95 ms"]
    lines = [
        [
            *row.group,
            str(row.calls),
            str(row.failed),
            f"{row.cost_usd:.6f}",
            str(row.input_tokens),
            str(row.output_tokens),
            str(row.reasoning_tokens),
            "-" if row.p50_latency_ms is None else str(row.p50_latency_ms),
            "-" if row.p95_latency_ms is None else str(row.p95_latency_ms),
        ]
        for row in rows
    ]
    total = sum(row.cost_usd for row in rows)
    widths = [max(len(cell) for cell in column) for column in zip(header, *lines, strict=True)]

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    body = [fmt(header), fmt(["-" * w for w in widths]), *(fmt(line) for line in lines)]
    return "\n".join([*body, f"\ntotal: ${total:.6f} over {sum(r.calls for r in rows)} calls"])


# A call outlives every timeout and retry long before this; a reservation still open is stuck.
STALE_AFTER = timedelta(minutes=30)


async def stale(session: AsyncSession, older_than: timedelta = STALE_AFTER) -> int:
    count = await session.scalar(
        select(func.count()).where(
            LlmCall.status == "reserved", LlmCall.created_at < func.now() - older_than
        )
    )
    return int(count or 0)


async def collect(
    session: AsyncSession, *, days: int, by: Sequence[str], expire: bool
) -> tuple[list[SpendRow], int]:
    """The report's rows and how many reservations are still stuck. Expiring comes first, so the
    rows include the calls it just closed (Codex review on PR 24)."""
    if expire:
        await expire_stale(session, older_than=STALE_AFTER)
        await session.commit()
    rows = await spend(session, since=utc_today() - timedelta(days=days - 1), by=by)
    return rows, await stale(session)


async def main(days: int, by: Sequence[str], expire: bool) -> None:
    engine = create_engine(get_settings())
    try:
        async with create_sessionmaker(engine)() as session:
            rows, stuck = await collect(session, days=days, by=by, expire=expire)
    finally:
        await engine.dispose()
    print(render(rows, by) if rows else f"No settled model calls in the last {days} days.")
    if stuck:
        print(f"\n{stuck} reservations open for over 30 minutes: rerun with --expire-stale.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model spend from the llm_call table.")
    parser.add_argument("--days", type=int, default=7, help="days back from today (UTC)")
    parser.add_argument("--by", default="day,model", help=f"comma-separated: {', '.join(GROUPS)}")
    parser.add_argument(
        "--expire-stale",
        action="store_true",
        help="close reservations open for over 30 minutes (keeping their reserved cost)",
    )
    args = parser.parse_args()
    by = [name.strip() for name in args.by.split(",") if name.strip()]
    if not by or any(name not in GROUPS for name in by):
        parser.error(f"--by takes {', '.join(GROUPS)}")
    asyncio.run(main(args.days, by, args.expire_stale))

"""The agent step limit, the same for every framework: an agent run makes at most N model calls.

Call k of a run (counting from 1) is offered the tools while k < N. Call N is made without tools,
so the model writes a real final answer instead of asking for another tool, and the result is
flagged `step_limit_reached`. The count is per run, not per thread. An adapter counts its own
framework's model calls, asks StepLimit what to offer, and logs with `log_reached` once the final
call has answered (app/llm/langchain/steps.py for LangChain).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

__all__ = ["DEFAULT_MAX_MODEL_CALLS", "STEP_LIMIT_REACHED", "StepLimit", "log_reached"]

log = structlog.get_logger()
# The log event and the key of the result's flag.
STEP_LIMIT_REACHED = "step_limit_reached"
# The old default of 25 LangGraph steps allowed ceil(25 / 2) = 13 model calls.
DEFAULT_MAX_MODEL_CALLS = 13


@dataclass(frozen=True, slots=True)
class StepLimit:
    max_model_calls: int

    def __post_init__(self) -> None:
        if self.max_model_calls < 1:
            raise ValueError(f"The model-call limit must be at least 1, not {self.max_model_calls}")

    def is_final(self, calls_made: int) -> bool:
        """True when the call about to be made (call `calls_made + 1`) is the last allowed one."""
        return calls_made + 1 >= self.max_model_calls

    def offers_tools(self, calls_made: int) -> bool:
        return not self.is_final(calls_made)


def log_reached(limit: StepLimit) -> None:
    """Log that a run ended on its forced final answer (a warning: the answer was cut short)."""
    log.warning(STEP_LIMIT_REACHED, max_model_calls=limit.max_model_calls)

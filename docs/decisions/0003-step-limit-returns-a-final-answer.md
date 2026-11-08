# 0003. Both frameworks return a final answer at the step limit

Status: Proposed. The rule is decided (founder, 6 Nov 2026); the unit and the off-by-one are not, and the founder has still to confirm the recommendations on [#34](https://github.com/olaheavy-dev/house-template-fastapi/issues/34).

## Context

When an agent reaches its limit, the LangChain gateway raises LangGraph's `GraphRecursionError` and the API answers 422 (`StepLimitReached`). MAF's tool loop instead stops calling tools and asks the model for a final text answer. A project should get the same API behaviour whichever framework it picks.

## Decision (decided)

At the step limit: tools stop being offered, the model is asked once for a final answer, the response carries a neutral flag saying the limit was reached, and the call is logged with `step_limit_reached`. The rule and the flag live in the framework-neutral core; each adapter applies them. The final-answer call is budgeted and recorded like any other call.

Verified in the spike ([#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32), item 8): a `wrap_model_call` middleware that calls `handler(request.override(tools=[]))` on the last allowed call gives a real model answer. LangChain's built-in `ModelCallLimitMiddleware(exit_behavior="end")` returns a fixed message instead, so it is not used. `recursion_limit` stays as a backstop at 2N+1 or more. The docs do not name this as the step-limit recipe; it is built from documented parts.

## Open questions

Recommendations are from [#34](https://github.com/olaheavy-dev/house-template-fastapi/issues/34); none is confirmed.

1. **What a step is.** Today `AGENT_MAX_STEPS` counts LangGraph steps (default 25, about 12 model calls). Counting model calls with the same default would roughly double the most one run can spend. Recommendation: count model calls and lower the default to 12.
2. **N or N+1 calls.** Recommendation: exactly N model calls in total, the last being the final answer, so the cap is predictable and the budget reservation can be worked out in advance.
3. **Where the count lives.** Recommendation: our own middleware in the core, not each framework's own limit, so both count the same. MAF's `max_iterations` (default 40 in 1.24.0) is set explicitly to the same N.

## What exists today

The 422 `StepLimitReached` (core since [#66](https://github.com/olaheavy-dev/house-template-fastapi/pull/66)) and `AGENT_MAX_STEPS` are what ships. [#34](https://github.com/olaheavy-dev/house-template-fastapi/issues/34) (LangChain half, planned for 0.8.0 after T5b) and [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) sub-task 5b (MAF half and the shared test) are not built.

## Consequences

- Breaking for projects that relied on the 422; goes in the 0.8.0 migration notes.
- One shared test, run against both frameworks, with the expected call count worked out by hand.

## Alternatives considered

- Keep the 422 on LangChain: the two frameworks would differ. Rejected by the founder's decision.
- Use the built-in `ModelCallLimitMiddleware`: stops cleanly but gives a fixed message, not an answer.

## Sources

- [#34](https://github.com/olaheavy-dev/house-template-fastapi/issues/34), [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35), spike [#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32).
- LangChain built-in middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in
- LangChain custom middleware, dynamic tools: https://docs.langchain.com/oss/python/langchain/middleware/custom
- LangGraph recursion limit: https://docs.langchain.com/oss/python/langgraph/graph-api

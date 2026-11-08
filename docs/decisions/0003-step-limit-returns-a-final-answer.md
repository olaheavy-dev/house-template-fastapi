# 0003. Both frameworks return a final answer at the step limit

Status: Proposed. The rule is decided (founder, 6 Nov 2026). The unit of the limit and the off-by-one are not, and the founder has still to confirm the recommendations on [#34](https://github.com/olaheavy-dev/house-template-fastapi/issues/34).

## Context

When an agent reaches its limit, LangGraph raises `GraphRecursionError`, and the template answers it with a 422: `core.step_limit_response` (message "The assistant couldn't finish this within its step limit.", logged as `agent_step_limit`; wired in `langchain/errors.py`). `StepLimitReached` is defined in the core, but nothing raises it yet ([#66](https://github.com/olaheavy-dev/house-template-fastapi/pull/66) upgrade notes). MAF's tool loop instead stops calling tools and asks the model for a final text answer. A project should get the same API behaviour whichever framework it picks.

## Decision (decided)

At the step limit: tools stop being offered, the model writes one final answer, the response carries a neutral flag saying the limit was reached, and the call is logged with `step_limit_reached`. How many model calls that takes in total is open question 2 below. The rule and the flag live in the framework-neutral core; each adapter applies them. The final-answer call is budgeted and recorded like any other call.

`step_limit_reached` is a new log event (today's is `agent_step_limit`). The #33 plan promised unchanged log names, so this is a deliberate change for 0.8.0 and goes in its migration notes.

Verified in the spike ([#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32), item 8): a `wrap_model_call` middleware that calls `handler(request.override(tools=[]))` on the last allowed call gives a real model answer. LangChain's built-in `ModelCallLimitMiddleware(exit_behavior="end")` returns a fixed message instead, so it is not used. `recursion_limit` stays as a backstop at 2N+1 or more. The docs do not name this as the step-limit recipe; it is built from documented parts.

## Open questions

Recommendations are from [#34](https://github.com/olaheavy-dev/house-template-fastapi/issues/34); none is confirmed.

1. **What a step is.** Today `AGENT_MAX_STEPS` is LangGraph's `recursion_limit`. Measured on 8 Nov 2026 with the locked versions (langchain 1.4.7, langchain-core 1.7.3, langgraph 1.2.19), `create_agent` and a model that always asks for a tool: the default 25 allows up to **13 model calls** before the error (limit 6 gives 3, as in spike #32; the rule is ceil(limit/2)). Counting model calls but keeping 25 would let one looping run make up to 25 calls, roughly doubling the worst-case spend. Recommendation: count model calls and set the default to **13**, which keeps the worst case equal (12 would lower it by one call, about 8%); say so in the migration notes.
2. **N or N+1 calls.** Is the final answer the Nth call, or an extra call after N? Recommendation: exactly N model calls in total, the last being the final answer, so the cap is predictable and the budget reservation can be worked out in advance.
3. **Where the count lives.** Recommendation: our own middleware in the core, not each framework's own limit, so both count the same. MAF's `max_iterations` (default 40 in 1.24.0, per spike [#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32) items 3 and 8) would be set explicitly as a backstop; how it maps to model calls is unverified.

## What exists today

The 422 described above and `AGENT_MAX_STEPS` are what ships. [#34](https://github.com/olaheavy-dev/house-template-fastapi/issues/34) (LangChain half, planned for 0.8.0 after T5b) and [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) sub-task 5b (MAF half and the shared test) are not built.

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

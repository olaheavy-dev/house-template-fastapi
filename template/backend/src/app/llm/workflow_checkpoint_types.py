"""The types your MAF workflow checkpoints may hold, besides MAF's own safe set (yours to edit, like
prices.py).

A checkpoint is saved as JSON, with every value that is not JSON-native pickled inside it, and read
back with MAF's restricted unpickler (app/llm/maf/checkpoints.py). That rebuilds only MAF's built-in
safe set (primitives, dates and times, UUIDs, decimals, sets and the like), MAF's and openai's own
types, and the types listed in ALLOWED; a checkpoint that holds anything else is refused when it is
saved, so you find out before a workflow needs to resume from it.

List EVERY type your workflow passes between executors or keeps in its state or executor state
(`on_checkpoint_save`): dataclasses, pydantic models, enums, request types for human-in-the-loop
steps. Each entry is "module:qualname", the dotted module path, a colon, and the class's
`__qualname__`:

    # app/workflows/models.py defines `class ComputeTask` and `class Approval`
    ALLOWED: frozenset[str] = frozenset(
        {"app.workflows.models:ComputeTask", "app.workflows.models:Approval"}
    )

List only plain data types whose construction has no side effects. The allow-list narrows what a
stored row can build, but MAF's own documentation says it "is **not** a security boundary": the
real protection is that only the app writes checkpoint rows, in its own database. Keep this file
framework-free (the core may import it).
"""

from __future__ import annotations

__all__ = ["ALLOWED"]

ALLOWED: frozenset[str] = frozenset()

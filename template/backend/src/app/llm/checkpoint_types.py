"""The classes your agent threads may hold, besides LangGraph's own safe ones (yours to edit, like
prices.py).

A thread is saved as bytes and rebuilt when it is loaded. The checkpointer rebuilds only what is on
LangGraph's safe list (messages and their usage metadata, `Send`, `Command`, `Interrupt`, dates and
times, UUIDs, decimals, sets, deques, IP addresses, paths, compiled regular expressions and a few
more) and the classes listed in ALLOWED. Any other saved object comes back as its plain data, a
dict or a list, and LangGraph logs "Blocked deserialization of <module>.<name>" once per class and
process. Without this, a row that names a function would have it imported and called on load
(GHSA-g48c-2wqr-h844); that needs someone who can already write to your database, but it costs
nothing to close.

If your graph keeps your own class in its state (a pydantic model, a dataclass, an Enum), list it
here, or the next step that reads it gets a dict instead of an object:

    from app.answers import Answer

    ALLOWED: tuple[type, ...] = (Answer,)

List only plain data classes whose construction has no side effects. A pydantic model is built
without its validators when a saved one is loaded (LangGraph uses model_construct), so a row with
the wrong shape comes back as an object with the wrong fields. A class defined inside a function or
another class cannot be listed: it is rebuilt by its module and name.

This module imports nothing from the model frameworks, so a class listed here can be imported by
the framework-neutral core. Threads saved before this existed need nothing: everything LangGraph
saves for a normal tool-calling agent is on the safe list.
"""

from __future__ import annotations

ALLOWED: tuple[type, ...] = ()

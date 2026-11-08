"""The classes your agent threads may hold, besides LangGraph's own safe ones (yours to edit, like
prices.py).

A thread is saved as bytes and rebuilt when it is loaded. The checkpointer rebuilds only what is on
LangGraph's safe list (messages and their usage metadata, `Send`, `Command`, `Interrupt`, dates and
times, UUIDs, decimals, sets, deques, IP addresses, paths, compiled regular expressions and a few
more) and the classes listed in ALLOWED. Any other saved object comes back as its plain data (a
dict, a list, or an enum's value), and LangGraph logs "Blocked deserialization of <module>.<name>"
once per class and process. Without this, a row that names a function would have it imported and
called on load (GHSA-g48c-2wqr-h844); that needs someone who can already write to your database, but
it costs nothing to close.

List EVERY class your graph keeps in its state before you deploy: pydantic models, dataclasses,
Enums (also nested ones, as in `list[Answer]`), a `response_format` model's `structured_response`,
and the artifacts of `content_and_artifact` tools. With the LANGGRAPH_STRICT_MSGPACK environment
variable on, `StateGraph.compile` would derive these from the state types for you; this explicit
setting does not, so nothing is listed until you list it. An unlisted class that is saved again
(the thread runs another turn) is saved as its plain data for good, so rolling back does not
restore the object.

    from app.answers import Answer

    ALLOWED: tuple[type, ...] = (Answer,)

List only plain data classes whose construction has no side effects. A listed pydantic model is
built with its normal constructor, validators included; only if that raises does LangGraph fall
back to `model_construct`, which skips validation, so a row of the wrong shape can come back as an
object with the wrong fields. A class defined inside a function or another class cannot be listed:
it is rebuilt by its module and name. Keep the classes in modules that import cheaply and do not
import app.llm.langchain.checkpoints. Keep this file framework-free so the core may import it.

Threads saved before this existed need nothing for a normal tool-calling agent: everything
LangGraph saves for it is on the safe list.
"""

from __future__ import annotations

ALLOWED: tuple[type, ...] = ()

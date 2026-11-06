"""Test fixtures: the models the recorded cassettes were made with. They are pinned so replay stays
the same, they are NOT defaults (the template picks no model), and both are priced in
app/llm/prices.py.

To record with another model, change CHAT_MODEL (and EMBEDDING_MODEL) here, then run
`make record-llm`: replay matches requests on their body, so the cassettes and these ids must agree.
`make eval` is different: it uses your own LLM_MODEL and EMBEDDING_MODEL (see tests/conftest.py)."""

CHAT_MODEL = "openai:gpt-6-luna"
EMBEDDING_MODEL = "openai:text-embedding-3-small"

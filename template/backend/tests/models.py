"""Test fixtures: the models the recorded cassettes were made with. They are pinned so replay stays
the same, they are NOT defaults (the template picks no model), and both are priced in
app/llm/prices.py. `make record-llm` and `make eval` use the project's own LLM_MODEL and
EMBEDDING_MODEL instead (see tests/conftest.py)."""

CHAT_MODEL = "openai:gpt-6-luna"
EMBEDDING_MODEL = "openai:text-embedding-3-small"

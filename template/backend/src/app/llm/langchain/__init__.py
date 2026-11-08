"""The LangChain adapter: converts what LangChain reports (usage, model errors) into the core's
neutral types, drives the core's budget Meter from LangChain's callbacks (callbacks.py), and wraps
the core's metered embeddings as LangChain's Embeddings class (embeddings.py)."""

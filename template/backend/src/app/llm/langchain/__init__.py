"""The LangChain adapter: converts what LangChain reports (usage, model errors) into the core's
neutral types, drives the core's budget Meter from LangChain's callbacks (callbacks.py), and wraps
the core's metered embeddings as LangChain's Embeddings class (embeddings.py), gives the core's
prompt its LangChain call config (prompts.py), builds the chat model, agents and call config
(gateway.py: Gateway, get_gateway; product code imports GatewayDep from app.llm), applies the
core's step limit to create_agent (steps.py) and holds LangGraph's Postgres checkpointer
(checkpoints.py), and reads a stored thread's messages for the context report (context.py)."""

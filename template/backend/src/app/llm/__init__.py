"""The model gateway: the only package that configures or calls a language model provider.

gateway.py builds the model, prices.py holds the project's price tables, and prompts live in
core/prompts.py (the prompt and its hash) and langchain/prompts.py (load_prompt and config()).
core/ prices calls, caps daily spend (budget.py, bounds.py), records metrics, classifies failures
and meters batched OpenAI embeddings (embeddings.py) without any framework; langchain/ converts
LangChain's data, drives the budget from its callbacks and wraps the embeddings as LangChain's
Embeddings class (embeddings.py). See gateway.py for how to call the model.
"""

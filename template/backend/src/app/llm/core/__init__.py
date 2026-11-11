"""The framework-neutral part of the model gateway: what a call used (usage.py), what it costs
(prices.py), which key and price a configured model has (registry.py), how a failed call is
classified and answered (errors.py), the agent step limit (steps.py), how much a request may cost
before it is sent (bounds.py), the daily budget and its llm_call rows (budget.py: an adapter
drives its Meter), the llm.* metrics (metrics.py), and embeddings batched and metered on a Meter
(embeddings.py) with OpenAI's call behind them (openai_embeddings.py), the prompt and its version
hash (prompts.py), personal-data redaction (redaction.py, with the string walkers and the restorer
of the originals in redaction_data.py; the project's detector list is app/llm/redaction_rules.py)
and the spend report on the llm_call table (report.py). Nothing here
imports a model framework or a provider SDK except openai_embeddings.py, which imports openai
(lint-imports enforces it, llm-contracts.toml); each framework's adapter converts its own data to
these types at its edge (app/llm/langchain/ for LangChain).
"""

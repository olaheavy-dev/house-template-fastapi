"""The framework-neutral part of the model gateway: what a call used (usage.py), what it costs
(prices.py), which key and price a configured model has (registry.py), how a failed call is
classified and answered (errors.py), how much a request may cost before it is sent (bounds.py),
the daily budget and its llm_call rows (budget.py: an adapter drives its Meter) and the llm.*
metrics (metrics.py). Nothing here imports a model framework or a provider SDK (lint-imports
enforces it, llm-contracts.toml); each framework's adapter converts its own data to these types at
its edge (app/llm/langchain/ for LangChain).
"""

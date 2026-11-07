"""The framework-neutral part of the model gateway: what a call used (usage.py), what it costs
(prices.py), which key and price a configured model has (registry.py) and how a failed call is
classified and answered (errors.py). Nothing here imports a
model framework or a provider SDK (lint-imports enforces it, llm-contracts.toml); each framework's
adapter converts its own data to these types at its edge (app/llm/langchain/ for LangChain).
"""

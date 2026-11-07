"""The framework-neutral part of the model gateway: what a call used (usage.py), what it costs
(prices.py) and which key and price a configured model has (registry.py). Nothing here imports a
model framework or a provider SDK (lint-imports enforces it, llm-contracts.toml); each framework's
adapter converts its own data to these types at its edge (app/llm/langchain/ for LangChain).
"""

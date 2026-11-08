"""The model gateway: the only package that configures or calls a language model provider.

gateway.py builds the model, prices.py holds the project's price tables and prompts.py loads
versioned prompts. core/ prices calls, caps daily spend (budget.py, bounds.py), records metrics and
classifies failures without any framework; langchain/ converts LangChain's data and drives the
budget from its callbacks. See gateway.py for how to call the model.
"""

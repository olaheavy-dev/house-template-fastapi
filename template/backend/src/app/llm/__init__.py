"""The model gateway: the only package that configures or calls a language model provider.

gateway.py builds the model, budget.py caps daily spend, prices.py holds the project's price tables
(core/ prices calls and classifies their failures without any framework) and prompts.py loads
versioned prompts; langchain/ converts LangChain's data. See gateway.py for how to call the model.
"""

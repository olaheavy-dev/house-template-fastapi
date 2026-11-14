"""Microsoft Agent Framework (MAF) adapter of the model gateway (#35, docs/decisions/0001).

The only package that imports `agent_framework` (and its OpenAI and Anthropic packages):
lint-imports and a test enforce it. gateway.py builds the chat client and `GatewayDep`;
middleware.py is the budget and redaction.py the personal-data redaction, both MAF chat
middleware; usage.py and errors.py turn MAF's usage and errors into the core's neutral ones
(app/llm/core/); embeddings.py meters OpenAI's embeddings under the same budget; prompts.py loads
a prompt with its version; context.py reads a stored session for the context report and keeps
each answer's usage (UsageHistoryProvider). Nothing here is imported by the core.
"""

# Prompts

One Markdown file per prompt, loaded with `app.llm.prompts.load_prompt("<file name without .md>")`.
The file's content hash is the prompt's version: it is stored with each call's cost and tagged on its
LangSmith trace. Change a prompt in a pull request like any other code, and re-record the cassettes of
the tests that send it (`make record-llm`).

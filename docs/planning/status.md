# Status

The handover for this repository. Update it in any pull request that changes what is done, open or next, a decision, a plan or the backlog, and overwrite it at the end of each session (house rules, agent-workflows v0.6.0).

Last updated: 13 Nov 2026.

## Where things stand

- **Released:** v0.10.1 (13 Nov 2026). 0.10.0 added the `llm: maf` answer (Microsoft Agent Framework, epic #35, PRs #112 and #118 to #123); 0.10.1 is a dependency update (#124).
- **Open pull requests:** #131, #46 PR 1, the context-window report (branch `feat/context-window-report`), in review.
- **#35 (MAF):** closed on 13 Nov 2026, with every planned PR merged. Its leftover is #127.

## Next

1. **#46: context-window report and free trimming (no summarising).** The docs and source check is done, and the decisions are recorded in #46's comments (13 Nov 2026):
   - the window size is a field on the price entry;
   - tokens are counted from the last call's usage plus an estimate, calling the provider's count endpoint only near a threshold;
   - trimming always keeps the user's first message.

   PR 1's own decisions were made on 13 Nov 2026 and are recorded in #46:
   - 100% is where calls start failing: the nearer of the window less `LLM_MAX_TOKENS` and, for a price with `max_prompt_tokens`, the budget's byte bound against it (refined after the pre-push review);
   - the provider's count endpoint waits for PR 2;
   - MAF keeps each call's usage on the stored answer;
   - the budget records the bound it checked on each stored answer (both frameworks), and the report's bound starts from it, rather than only rebuilding what an agent sends (chosen in Codex round 5 on #131: "Redesign now, in #131");
   - there is no HTTP route; the report is a gateway method.

   Plan:
   - PR 1: the core and the report. **In review.** It adds window sizes on the price entries, `gateway.context_report()` under both frameworks, and the trim policy (not applied yet).
   - PR 2: trimming and refusal. It applies `plan_trim` at 80% down to 50%, calls the count endpoint near the threshold (also for the budget's prompt cap, so OpenAI conversations stop being refused early), and refuses at 100%. It also measures MAF's double count of tool results (review P3-5).
   - PR 3: live evals of answer quality after a trim.
2. Follow-ups the owner may want next: #127, a LangChain `delete_user_data` to match MAF's (MAF's removes sessions and checkpoints). #117: adopt the shared agent workflow, including an `AGENTS.md` that points agents to this file (the repo has none yet).

## Blockers

None.

## Owner to-dos

- Review #131 (#46 PR 1) and say `merge 131` when it's ready; then say go (or not yet) for PR 2.
- agent-workflows #15 (issue) proposes that every repository keeps a status file. Checked 13 Nov: agent-workflows main doesn't cover it yet. For now only the adoption issues (#117 here, nextjs #21) require the file. #13 has merged, so #15 is unblocked. Say whether to write its pull request. It changes the house rules, so it's a founder-only merge.

## Working notes for agents

- **Check before every push:** a fresh render of each framework, plus the root suite: `uv run pytest` here, and `make check` inside rendered projects.
- **Live evals** run in a rendered project, with `LLM_EVAL=1`, a small `LLM_DAILY_BUDGET_USD` and the provider keys in the environment.
  - **Models:** any model priced in `app/llm/prices.py`. The ids pinned for recording in the rendered project's `backend/tests/models.py` (`CHAT_MODEL` as `LLM_MODEL`, and `EMBEDDING_MODEL`) work. They aren't defaults.
  - **Keys:** never print or commit them. A developer's local `template/backend/.env` is git-ignored.
- **Releases:** merge the release-please PR only on the owner's explicit "merge N". Merging needs a branch that is up to date with main, and release-please does not refresh its PR for `chore` commits, so use `gh pr update-branch N`.

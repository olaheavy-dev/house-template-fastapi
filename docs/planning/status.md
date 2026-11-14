# Status

The handover for this repository. Update it in any pull request that changes what is done, open or next, a decision, a plan or the backlog, and overwrite it at the end of each session (house rules, agent-workflows v0.6.0).

Last updated: 14 Nov 2026.

## Where things stand

- **Released:** v0.10.1 (13 Nov 2026). 0.10.0 added the `llm: maf` answer (Microsoft Agent Framework, epic #35, PRs #112 and #118 to #123); 0.10.1 is a dependency update (#124).
- **Merged, not yet released:** #131, #46 PR 1, the context-window report (14 Nov 2026). Release PR #133 (0.11.0) carries it.
- **Open pull requests:** #133, the 0.11.0 release; and this status update.
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
   - PR 1: the core and the report. **Merged (#131, 14 Nov 2026).** It adds window sizes on the price entries, `gateway.context_report()` under both frameworks, the bound the budget checked recorded on each stored answer, and the trim policy (not applied yet). It took six Codex rounds; rounds 1 to 5 were fixed in the PR (the fifth by the redesign above), and round 6 was deferred by the owner ("Follow-up to PR 2, merge now") as #132.
   - PR 2: trimming and refusal. It applies `plan_trim` at 80% down to 50%, calls the count endpoint near the threshold (also for the budget's prompt cap, so OpenAI conversations stop being refused early), and refuses at 100%. It also measures MAF's double count of tool results (review P3-5) and fixes #132 (after a step-limited LangChain run, the token figure leaves out the tool definitions the next run sends again).
   - PR 3: live evals of answer quality after a trim.
2. Follow-ups the owner may want next: #127, a LangChain `delete_user_data` to match MAF's (MAF's removes sessions and checkpoints). #117: adopt the shared agent workflow, including an `AGENTS.md` that points agents to this file (the repo has none yet).

## Blockers

None.

## Owner to-dos

- Say go (or not yet) for #46 PR 2 (trimming at 80% down to 50%, the provider's count near the threshold, refusal at 100%, and #132).
- Say `merge 133` when you want 0.11.0 (#46 PR 1) released.
- agent-workflows #16 (process issue, 13 Nov): the six Codex rounds on #131 came from one cause, a figure that re-derived another code path's figure. It proposes testing such figures against the real path from the start. Review it in the weekly process check.
- agent-workflows #15 (issue) proposes that every repository keeps a status file. Checked 13 Nov: agent-workflows main doesn't cover it yet. For now only the adoption issues (#117 here, nextjs #21) require the file. #13 has merged, so #15 is unblocked. Say whether to write its pull request. It changes the house rules, so it's a founder-only merge.

## Working notes for agents

- **Check before every push:** a fresh render of each framework, plus the root suite: `uv run pytest` here, and `make check` inside rendered projects.
- **Live evals** run in a rendered project, with `LLM_EVAL=1`, a small `LLM_DAILY_BUDGET_USD` and the provider keys in the environment.
  - **Models:** any model priced in `app/llm/prices.py`. The ids pinned for recording in the rendered project's `backend/tests/models.py` (`CHAT_MODEL` as `LLM_MODEL`, and `EMBEDDING_MODEL`) work. They aren't defaults.
  - **Keys:** never print or commit them. A developer's local `template/backend/.env` is git-ignored.
- **Releases:** merge the release-please PR only on the owner's explicit "merge N". Merging needs a branch that is up to date with main, and release-please does not refresh its PR for `chore` commits, so use `gh pr update-branch N`.

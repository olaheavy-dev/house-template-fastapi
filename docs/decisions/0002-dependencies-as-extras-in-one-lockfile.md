# 0002. Dependencies are chosen by extras in one lockfile, with conflicts

Status: Accepted (founder decision, 6 Nov 2026, recorded on [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30) and [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)).

## Context

A project that uses LangChain should not install MAF, and the other way round. There are two ways to do it: template `pyproject.toml` itself so each project gets different dependency lines, or keep optional extras in one plain `pyproject.toml` and let the project install the one it wants.

## Decision

Dependencies are optional extras in the one plain `backend/pyproject.toml` (`db`, `langchain`, `storage` exist today; `maf` is planned in [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)). `langchain` and `maf` are declared as conflicting, so they lock together in one `uv.lock` but cannot be installed together. Make targets and the image install only the extras the project answered for.

## What the spike found

[Spike #32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32), 6 Nov 2026, with uv 0.11.14:

- `uv lock` with `conflicts = [[{ extra = "langchain" }, { extra = "maf" }]]` resolves one lockfile. `uv sync --locked --extra maf` installs only the MAF packages, and `--extra langchain` swaps them for LangChain.
- Both extras together, or `--all-extras`, fail with "Extras `langchain` and `maf` are incompatible with the declared conflicts". So CI and make targets must name one extra.
- Dependabot: GitHub's documentation lists uv but does not mention extras; dependabot-core's source reads `project.optional-dependencies`. Whether Dependabot can re-lock a project that has `conflicts` is **unverified**. Check on the first Dependabot run, and keep `uv lock --check` in CI.
- `copier update` **deletes** files that a changed answer stops generating, even edited ones, with no conflict marker. Commit before updating.
- Pre-release `agent-framework-anthropic` needs an exact `==` pin and moves with core and openai; whether Dependabot proposes beta-to-beta updates is unverified.

## Consequences

- Dependabot still sees one `pyproject.toml` (the reason for not templating it).
- Anyone installing the backend by hand must pass `--extra` (the README's Upgrading section already warns of this for `db`, `langchain` and `storage`).
- Switching framework later is a lockfile change plus code, not a template re-render of dependencies.

## Alternatives considered

- Template `pyproject.toml` per answer: the file would no longer be plain, and the template's own tests and Dependabot would not read it as one. Not chosen (founder decision).
- Separate lockfiles per framework: more files to keep current. Not discussed in the issues; listed for completeness.

## Sources

- Spike findings: [#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32) (comments).
- uv, conflicting dependencies: https://docs.astral.sh/uv/concepts/projects/config/
- Plan: [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30), [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35).

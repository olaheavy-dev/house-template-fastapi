# 0006. Personal data is replaced with reversible placeholders in the neutral core before any provider call

Status: Accepted. The founder accepts the design and the eight answers recorded in issue [#78](https://github.com/olaheavy-dev/house-template-fastapi/issues/78) on 9 Nov 2026; this text was drafted by an agent and is merged by the founder. It builds on [0001](0001-framework-neutral-core-with-adapters.md) (a core with adapters) and [0005](0005-llm-answer-and-its-migrations.md) (the code exists only with `llm: langchain`). This record is the rule; the first pull request adds only the core and the project's detector list, and wires nothing. The adapters, settings, ledger column and trace masking follow in later pull requests of #78, so until then no text is changed on its way to a provider.

## Context

Text sent to a model provider can hold personal data: a user's email, phone number, bank details or national insurance number, typed into a question or found in an uploaded document. What the gateway stores today holds no message text (`llm_call` has counts, cost, `feature` and an opaque `user_ref`, and provider error text never reaches logs or clients). The text that leaves for the provider, the LangSmith traces of it and recorded cassettes are the places it can be seen. A downstream project (HomeScout, decision 0025) already requires redaction before text is sent.

LangChain callbacks observe but cannot change messages, so removing values needs a real interception point inside each framework. LangChain's own `PIIMiddleware` works only inside `create_agent`, belongs to one framework and knows few kinds, so it is neither neutral nor enough.

What the official sources say (fetched 9 Nov 2026; nothing beyond them is claimed):

- Personal data is "any information relating to an identified or identifiable natural person". The ICO page also says "pseudonymisation is effectively only a security measure. It does not change the status of the data as personal data." Placeholders with a mapping kept elsewhere are pseudonymisation.
- Data minimisation (Article 5(1)(c)): personal data must be "adequate, relevant and limited to what is necessary". The ICO page does not mention redaction, and **no rule that requires redaction was found**; none is claimed. This record chooses redaction as a minimising measure, not as a legal duty.
- Special category data needs both an Article 6 lawful basis and an Article 9 condition (ICO). A pattern matcher cannot find it, so it is out of scope here.
- Whenever a controller uses a processor there must be a written contract (ICO, Article 28). Redaction does not replace it.
- OpenAI states API data is not used to train its models unless a customer opts in, and abuse-monitoring logs are kept for up to 30 days by default. Anthropic states that it is the data processor for the Claude API and points to its commercial data retention policy for standard retention. That policy says that for API users "we automatically delete inputs and outputs on our backend within 30 days of receipt or generation", with exceptions: a service with longer retention under the customer's control (for example the Files API), terms agreed otherwise (for example zero data retention), inputs and outputs retained for up to 2 years (and classification scores for up to 7 years) when they are flagged for Usage Policy enforcement, and retention required by law. No provider-side redaction feature was found, and it would arrive after the data has reached the provider.

## Decision

- **Where it lives.** `app/llm/core/redaction.py` has no framework import (the core import contract keeps it that way, [0001](0001-framework-neutral-core-with-adapters.md)) and uses only the standard library: no new package. The project's detector list is `app/llm/redaction_rules.py`, which is the project's like `prices.py`: it is not on the template-owned list, and a template update leaves edits to it alone. Both files exist only with `llm: langchain`: they sit under `app/llm`, which `llm: none` leaves out.
- **What it does.** `Redactor.redact(texts)` replaces matches across the whole outgoing message list with placeholders `[KIND_n]` and returns the texts, a mapping from placeholder to original and a count per kind. `restore(text, mapping)` and `StreamRestorer` put originals back into an answer or a stream. The scope for a request is set with `async with redaction_scope(known=..., allow=...)` and held in a context variable, never in call metadata (which a tracer records).
- **Default set (narrow, on by default).** EMAIL; PHONE (UK formats starting 01, 02, 03 or 07, and +44; not when more digits follow, after a space or hyphen); NINO (two letters, six digits, a final letter A to D, in upper case or in lower case straight after the digits; the prefix is not validated, so it errs towards removing); IBAN (GB layout, checksum not required); SORT_CODE (six digits, with hyphens, spaces or none) and ACCOUNT (eight digits, or two groups of four), each only with "sort code" or "account"/"acc" in the 30 characters before; CARD (13 to 19 digits in one run, or grouped 4-4-4-4 or Amex 4-6-5, passing the Luhn check; not one repeated digit); NHS (10 digits that pass modulus 11, with "NHS" in the 30 characters before); DOB (a date, with "date of birth", "DOB" or "born" in the 30 characters before). Every pattern must run in linear time. The kinds and the code are the project's to change.
- **Names and addresses only as exact values.** The app passes the user's own details as `known` (kind to values). They are matched ignoring case and runs of whitespace, not inside a longer word, and restored in the form the app gave. Names and addresses are not guessed by a name-finding package.
- **Overlaps are replaced as a union, never split.** Where a known value and a detected value overlap only in part (the known name "Priya" inside `priya.shah.1987@example.com`), the whole union is replaced, so passing in more never leaks more. The union takes the kind of its longest member (a known value on a tie), and a known value that covers the whole union keeps its own placeholder and form. An allowed value keeps a detected value only when it contains all of it; an allowed "Leeds" does not keep `priya@leeds.ac.uk`.
- **`known` beats `allow`.** `allow` lists values that must stay (a property's address, an agent's office number) and beats the detectors. A `known` value overlapping an allowed one is still replaced, and the event `redaction_allow_conflict` is logged with the kind only.
- **Checked input.** A kind must be capital letters, digits and underscores (up to 49 characters), so every placeholder written is one the restore step reads; a bad kind in a scope, a detector or a returned span is refused (the first two at set-up, the last fails closed). A detector span must lie inside the text and not be empty, or the call fails closed. A single string where a list is expected (`texts`, `allow`, a known value list) raises `TypeError` instead of being read one character at a time.
- **Numbers stay stable.** A placeholder's number is fixed by first appearance across the whole message list, and the same string always gets the same placeholder, so earlier messages keep their text as the history grows (the provider's prompt cache keeps matching). Numbers that already appear literally in the input are skipped, and a literal placeholder the model echoes back is never restored.
- **Mapping in memory only; checkpoints keep originals.** The mapping lives for one call and is never stored or logged. Checkpoints and sessions keep the original text, because storing redacted text would mean storing the mapping, which is the same personal data.
- **Always fail closed.** Any error in redaction (a detector that raises, a bug) raises `RedactionFailed`, a 503 problem ("nothing was sent"), logs `llm_redaction_failed` with the error type name only, and the call is not made. The original error is not chained, because it may hold the text. There is no setting to send anyway.
- **Placement.** Redaction runs on the final outgoing message list (user input, system prompt, tool results, retrieved chunks, history) before the bounds check and the budget reservation, so the reservation measures exactly what is sent. Two plug-in points, in later pull requests: a wrapper around the chat model in the LangChain adapter, and a chat middleware in the MAF adapter (with [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)). Text sent for embeddings is redacted one way, with no restore.
- **Restored on the way back:** answer text, tool-call arguments before the tool runs (the adapter restores the parsed argument values, not the raw JSON string: an original holding a quote or a backslash would make invalid JSON), structured-output fields and streamed text (a stream holds back text from an unclosed `[` for at most the longest placeholder, or until a character that cannot be in a placeholder).
- **Traces, logs and the ledger (later pull requests).** LangSmith traces are masked one way with the same rules. `llm_call` gets only a count of items removed (`redacted_count`, null when redaction is off), never a value or a mapping. Log events never carry a value: only a kind, a count or an error type name.
- **Placeholders.** `[KIND_n]` with capital letters, digits and underscores. In an answer, `restore` counts every placeholder-shaped string that is not in the mapping, including ones a model has mangled (another case, a space or hyphen for the underscore, spaces inside the brackets); it leaves them as they are. A stream gives the same text and the same count as `restore` on the joined text, whatever the chunking.

### What this does not claim

- It is **not anonymisation**. Text with placeholders can still be personal data, and the mapping is the key.
- It is **not complete**. It finds common formats and exact values. Names and addresses not passed as `known`, third parties named in a document, free-form formats (a NINO written without its last letter) and text inside images are missed. A NINO-shaped reference is replaced even if it is not one.
- It has **no special-category detection**.
- Matching is exact in its own way. A zero-width space inside a known value defeats the match, and a known value written with ß does not match the same word written with "SS". A detected value is restored exactly as written (`A@X.COM` and `a@x.com` get different placeholders); only known values match ignoring case.
- A phone number or card number followed by a space or hyphen and another digit is not recognised (to leave energy certificate references and grouped amounts alone), so `020 7946 0321 5` and two phone numbers separated only by a space are missed. A 13-digit meter number can pass the card check and be replaced.
- It is **not a replacement for a lawful basis, a privacy notice or a processor contract** with the provider.

## Consequences

- With no match in the text, a request is byte-identical to what it was before, so existing cassettes still match. A project whose prompts hold matching values sees placeholders once the adapters are wired; the release notes of that pull request say so.
- The first pull request changes no behaviour: it adds the core, the detector list and tests.
- Every product that shows answers must show the restored text; a placeholder reaching a user is a bug. The count of placeholders the model invented or mangled (`unrestored`) is available to log.
- A false negative (a name not passed as `known`) sends that value to the provider. A false positive replaces something harmless; the keyword gates, the allow-list and a must-not-change test text limit it.
- A downstream project must say how it uses this: the app passes account details as `known` and the property's values as `allow`, and its product wording never calls the result "anonymised" or implies a guarantee.
- `known` values are matched ignoring case and whitespace, so a short value (a first name) would also match inside other people's names where it stands alone. Pass full names.
- A placeholder typed literally in a later message (`[EMAIL_1]` written by a user) is skipped when numbering, which renumbers the values of earlier messages for that call and breaks the provider's prompt cache once. It is rare.
- Known values cost time in proportion to the number of values times the text size (200 values over 1 MiB took about 1.5 s); a project with thousands of values should measure before relying on it.

## Alternatives considered

| Option | In practice | Dependency | Verdict |
|---|---|---|---|
| Own rules in the neutral core | patterns plus checksum or keyword checks, exact match on known values; shared by both frameworks; misses names it is not told about | none | **Chosen** |
| Microsoft Presidio | adds name and location detection and UK recognisers; needs spaCy and a language model; its own site says it cannot find everything; would also remove an agent's or landlord's name that the app wants sent | presidio-analyzer and presidio-anonymizer, spaCy, a model | Not now; possible later, optional, behind the same `Detector` interface, with the founder's approval, only if the eval shows names being missed (Presidio's own site says "there is no guarantee that Presidio will find all sensitive information") |
| LangChain `PIIMiddleware` | agents only, one framework, few kinds, restoring not documented | already installed | No |
| A provider-side feature | none found; the data has already reached the provider | n/a | No |
| A cloud PII service | sends the data to another processor to guard it from a processor | a new outside service | No |
| A small local model | image size, memory, latency and its own evals, accuracy unproven | a model and a runtime | No |
| Do nothing | breaks the downstream rule that personal data is redacted before text is sent | none | No |
| Fail open (send on a redaction error) | fewer failed requests | none | No: founder decision, always blocked |
| Store redacted text in checkpoints | nothing personal in the checkpoint table | none | No: it needs the mapping stored, which is the same personal data |

## Sources

- Plan and founder decisions: [#78](https://github.com/olaheavy-dev/house-template-fastapi/issues/78).
- ICO, what is personal data (definition; pseudonymisation): https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/personal-information-what-is-it/what-is-personal-data/what-is-personal-data/
- ICO, data minimisation (Article 5(1)(c)): https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-protection-principles/a-guide-to-the-data-protection-principles/data-minimisation/
- ICO, rules on special category data: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/what-are-the-rules-on-special-category-data/
- ICO, when a contract with a processor is needed: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/contracts-and-liabilities-between-controllers-and-processors-multi/when-is-a-contract-needed-and-why-is-it-important/
- OpenAI, your data: https://developers.openai.com/api/docs/guides/your-data
- Anthropic, API and data retention (Anthropic is the data processor; points to the commercial policy): https://platform.claude.com/docs/en/manage-claude/api-and-data-retention
- Anthropic, commercial data retention policy (30-day deletion and its exceptions): https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data
- Presidio: https://presidio.dataprivacystack.org/
- Python `re` (possessive quantifiers and atomic groups, added in 3.11; fixed-width lookbehind): https://docs.python.org/3.14/library/re.html
- [0001](0001-framework-neutral-core-with-adapters.md), [0005](0005-llm-answer-and-its-migrations.md).

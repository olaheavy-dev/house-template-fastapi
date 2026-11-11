"""Values inside structured data keep their field name for detection (#78): a context-gated rule
(date of birth, account, sort code, NHS number) needs the word that names the value, and a value in
a tool call's arguments or in JSON text has that word as its key. JSON text is decoded before its
strings are matched, so a known value that holds a quote, a backslash or a non-ASCII character is
found in its decoded form, not hidden by escaping."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.llm.core.redaction import Redactor, redaction_scope
from app.llm.langchain import redaction
from tests import wire
from tests.evals.redaction_corpus import MUST_NOT_CHANGE

ARGS = {"dob": "12/03/1980", "account": "12345678", "NHS": "943 476 5919", "sort_code": "20-45-67"}
ADDRESS = '12 "Quoted" Row\\Flat é'
REDACTOR = Redactor(wire.DETECTORS)


pytestmark = pytest.mark.anyio


def sent(message: BaseMessage) -> BaseMessage:
    return redaction.prepare(REDACTOR, [message]).messages[0]


async def prepared(message: BaseMessage, known: dict[str, list[str]]) -> redaction._Prepared:
    async with redaction_scope(known=known):
        return redaction.prepare(REDACTOR, [message])


def assert_no_value(text: str) -> None:
    for value in ARGS.values():
        assert value not in text, value


# --- P1-1: the field name is context ----------------------------------------------------------


def test_the_core_uses_a_context_for_detection_only_and_replaces_only_the_value() -> None:
    result = REDACTOR.redact(["12/03/1980", "12/03/1980"], contexts=["date of birth: ", ""])
    assert result.texts == ("[DOB_1]", "12/03/1980")  # the same date with no context stays
    assert result.counts == {"DOB": 1}


def test_a_context_that_is_not_there_does_not_make_a_match() -> None:
    result = REDACTOR.redact(["12345678", "ref 12345678"], contexts=["order id: ", ""])
    assert result.texts == ("12345678", "ref 12345678")


def test_parsed_tool_call_arguments_in_the_history_are_redacted_by_field_name() -> None:
    message = AIMessage("", tool_calls=[{"name": "save", "args": dict(ARGS), "id": "1"}])
    out = sent(message)
    assert isinstance(out, AIMessage)
    assert_no_value(json.dumps(out.tool_calls))
    assert out.tool_calls[0]["args"] == {
        "dob": "[DOB_1]",
        "account": "[ACCOUNT_1]",
        "NHS": "[NHS_1]",
        "sort_code": "[SORT_CODE_1]",
    }


def test_nested_and_listed_arguments_use_the_nearest_field_name() -> None:
    args = {"person": {"date_of_birth": "12/03/1980", "bankDetails": {"account": "12345678"}}}
    message = AIMessage("", tool_calls=[{"name": "save", "args": args, "id": "1"}])
    out = sent(message)
    assert isinstance(out, AIMessage)
    got = out.tool_calls[0]["args"]
    assert got["person"]["date_of_birth"] == "[DOB_1]"
    assert got["person"]["bankDetails"]["account"] == "[ACCOUNT_1]"


def test_a_raw_json_argument_string_is_redacted_by_field_name_and_valid_json() -> None:
    raw = json.dumps(ARGS)
    call = {"id": "1", "type": "function", "function": {"name": "save", "arguments": raw}}
    out = sent(AIMessage("", additional_kwargs={"tool_calls": [call]}))
    arguments = out.additional_kwargs["tool_calls"][0]["function"]["arguments"]
    assert_no_value(arguments)
    assert json.loads(arguments)["dob"] == "[DOB_1]"
    legacy = sent(
        AIMessage("", additional_kwargs={"function_call": {"name": "s", "arguments": raw}})
    )
    assert_no_value(legacy.additional_kwargs["function_call"]["arguments"])


def test_cut_off_argument_text_is_redacted_by_the_name_in_front_of_the_value() -> None:
    cut = '{"dob": "12/03/1980", "account": "12345678", "nhs": "943 476 5919", "sort'
    message = AIMessage(
        "", invalid_tool_calls=[{"name": "s", "args": cut, "id": "1", "error": "e"}]
    )
    out = sent(message)
    assert isinstance(out, AIMessage)
    assert out.invalid_tool_calls[0]["args"] is not None
    assert_no_value(out.invalid_tool_calls[0]["args"])


def test_an_anthropic_tool_use_block_and_a_tool_result_block_are_redacted_by_field_name() -> None:
    block = {"type": "tool_use", "id": "t1", "name": "save", "input": dict(ARGS)}
    out = sent(AIMessage(content=[block]))
    assert_no_value(json.dumps(out.content))
    result = {"type": "tool_result", "tool_use_id": "t1", "content": json.dumps(ARGS)}
    out2 = sent(HumanMessage(content=[result]))
    assert_no_value(json.dumps(out2.content))


def test_json_text_in_a_tool_result_is_redacted_by_field_name_and_stays_json() -> None:
    out = sent(ToolMessage(json.dumps({"customer": dict(ARGS), "ok": True}), tool_call_id="1"))
    assert isinstance(out.content, str)
    assert_no_value(out.content)
    decoded = json.loads(out.content)
    assert decoded["ok"] is True
    assert decoded["customer"]["dob"] == "[DOB_1]"


def test_values_without_a_field_name_for_them_are_unchanged() -> None:
    plain = {"reference": "12345678", "date": "12/03/1980", "phone_label": "ring me", "n": 5}
    message = AIMessage("", tool_calls=[{"name": "save", "args": plain, "id": "1"}])
    out = sent(message)
    assert isinstance(out, AIMessage)
    assert out.tool_calls[0]["args"] == plain
    assert sent(ToolMessage(json.dumps(plain), tool_call_id="1")).content == json.dumps(plain)


@pytest.mark.parametrize("text", MUST_NOT_CHANGE)
def test_the_property_text_corpus_is_unchanged_inside_json_under_neutral_field_names(
    text: str,
) -> None:
    args = {"summary": text, "notes": [text]}
    out = sent(AIMessage("", tool_calls=[{"name": "n", "args": args, "id": "1"}]))
    assert isinstance(out, AIMessage)
    assert out.tool_calls[0]["args"] == args
    tool = sent(ToolMessage(json.dumps(args), tool_call_id="1"))
    assert tool.content == json.dumps(args)


# --- P1-2: JSON text is decoded before matching ------------------------------------------------


KNOWN = {"ADDRESS": [ADDRESS]}


@pytest.mark.parametrize("ensure_ascii", [True, False])
async def test_a_known_value_with_quotes_backslashes_and_unicode_is_found_in_json_text(
    ensure_ascii: bool,
) -> None:
    content = json.dumps(
        {"address": ADDRESS, "n": [1, {"deep": ADDRESS}]}, ensure_ascii=ensure_ascii
    )
    result = await prepared(ToolMessage(content, tool_call_id="1"), KNOWN)
    [message] = result.messages
    assert isinstance(message.content, str)
    assert "Quoted" not in message.content
    assert "Flat" not in message.content
    decoded = json.loads(message.content)
    assert decoded["address"] == "[ADDRESS_1]"
    assert decoded["n"][1]["deep"] == "[ADDRESS_1]"
    assert result.count == 2


async def test_json_inside_json_text_is_decoded_again() -> None:
    inner = json.dumps({"address": ADDRESS})
    content = json.dumps({"payload": inner})
    result = await prepared(ToolMessage(content, tool_call_id="1"), KNOWN)
    assert "Quoted" not in result.messages[0].content
    assert result.count == 1


async def test_a_known_value_in_json_fragments_inside_prose_is_found_in_its_escaped_form() -> None:
    text = "Result: " + json.dumps({"address": ADDRESS}) + " end"
    result = await prepared(HumanMessage(text), KNOWN)
    assert "Quoted" not in result.messages[0].content
    assert str(result.messages[0].content).startswith("Result: ")
    assert result.count == 1


async def test_a_known_value_in_plain_text_is_still_found_and_other_text_is_unchanged() -> None:
    text = f"I live at {ADDRESS}. Rent is £1,250 pcm."
    result = await prepared(HumanMessage(text), KNOWN)
    assert result.messages[0].content == "I live at [ADDRESS_1]. Rent is £1,250 pcm."
    untouched = await prepared(HumanMessage("Rent is £1,250 pcm, EPC band C."), KNOWN)
    assert untouched.messages[0].content == "Rent is £1,250 pcm, EPC band C."


def test_json_text_that_changes_nothing_is_sent_byte_for_byte() -> None:
    content = json.dumps({"a": 1, "b": ["x", "y"]}, indent=2)
    assert sent(ToolMessage(content, tool_call_id="1")).content == content


async def test_the_exact_examples_never_reach_the_wire_in_a_real_request() -> None:
    _, model = wire.wrapped("openai")
    history: list[BaseMessage] = [
        HumanMessage("save it"),
        AIMessage("", tool_calls=[{"name": "save", "args": dict(ARGS), "id": "c1"}]),
        ToolMessage(json.dumps({"address": ADDRESS, **ARGS}), tool_call_id="c1"),
    ]
    async with redaction_scope(known={"ADDRESS": [ADDRESS]}):
        with wire.captured(lambda _b: wire.openai_text("ok")) as request:
            await model.ainvoke(history)
    text = request.text
    assert_no_value(text)
    assert "Quoted" not in text
    assert "Flat" not in text
    assert "\\u00e9" not in text
    assert any(isinstance(b, dict) for b in request.json_bodies())


async def test_restoring_a_tool_call_returns_the_original_arguments_for_the_tool() -> None:
    from langchain_core.language_models import GenericFakeChatModel

    from app.llm.langchain.redaction import RedactingChatModel

    answer = AIMessage(
        "",
        tool_calls=[
            {"name": "save", "args": {"dob": "[DOB_1]", "account": "[ACCOUNT_1]"}, "id": "1"}
        ],
    )
    model = RedactingChatModel(GenericFakeChatModel(messages=iter([answer])), REDACTOR)
    history = [
        HumanMessage("save it"),
        AIMessage("", tool_calls=[{"name": "s", "args": dict(ARGS), "id": "0"}]),
    ]
    out = await model.ainvoke(history)
    assert out.tool_calls[0]["args"] == {"dob": ARGS["dob"], "account": ARGS["account"]}


# --- the same walk masks LangSmith traces --------------------------------------------------------


async def test_trace_masking_sees_field_names_and_decodes_json_text() -> None:
    from app.llm.langchain.redaction import masking_anonymizer

    anonymize = masking_anonymizer(REDACTOR)
    payload = {
        "inputs": {"args": dict(ARGS)},
        "outputs": {"content": json.dumps({"address": ADDRESS})},
    }
    async with redaction_scope(known=KNOWN):
        masked = anonymize(payload)
    text = json.dumps(masked)
    assert_no_value(text)
    assert "Quoted" not in text
    assert masked["inputs"]["args"]["dob"] == "[DOB_1]"
    assert json.loads(masked["outputs"]["content"])["address"] == "[ADDRESS_1]"


# --- the core: a known value is also matched in its JSON-escaped form -----------------------------


async def test_the_core_matches_a_known_value_written_as_json_escapes() -> None:
    escaped_ascii = json.dumps(ADDRESS)[1:-1]
    escaped_unicode = json.dumps(ADDRESS, ensure_ascii=False)[1:-1]
    assert escaped_ascii != escaped_unicode
    async with redaction_scope(known=KNOWN):
        result = REDACTOR.redact(
            [f"a {escaped_ascii} b", f"a {escaped_unicode} b", f"a {ADDRESS} b"]
        )
    assert result.texts == ("a [ADDRESS_1] b",) * 3
    assert result.mapping == {"[ADDRESS_1]": ADDRESS}


def test_a_match_that_starts_in_the_field_name_is_not_a_value_and_is_ignored() -> None:
    from app.llm.core.redaction import RegexDetector

    detector = RegexDetector("TOKEN", r"secret: \d+")
    redactor = Redactor([detector])
    # The pattern needs the field name AND the digits, so it begins in the context: not replaced.
    assert redactor.redact(["1234"], contexts=["secret: "]).texts == ("1234",)
    # The same text written in the value itself is a match inside the value.
    assert redactor.redact(["secret: 1234"], contexts=[""]).texts == ("[TOKEN_1]",)


def test_one_context_for_each_text_is_required() -> None:
    from app.llm.core.redaction import RedactionFailed

    with pytest.raises(RedactionFailed):
        REDACTOR.redact(["a", "b"], contexts=["only one"])


# --- the same kinds under other field names (schema.org, FHIR, common API names) ------------------

UK_DATE = "12/03/1980"
ISO_DATE = "1980-03-12"


def redacted_args(args: dict[str, Any]) -> Any:
    out = sent(AIMessage("", tool_calls=[{"name": "n", "args": args, "id": "1"}]))
    assert isinstance(out, AIMessage)
    return out.tool_calls[0]["args"]


def redacted_json(value: Any) -> Any:
    out = sent(ToolMessage(json.dumps(value), tool_call_id="1"))
    assert isinstance(out.content, str)
    return json.loads(out.content)


DOB_NAMES = [
    "dob", "DOB", "d.o.b", "birthDate", "birth_date", "birthdate", "BirthDate", "dateOfBirth",
    "date_of_birth", "date-of-birth", "patientBirthDate", "birthday",
]  # fmt: skip


@pytest.mark.parametrize("value", [UK_DATE, ISO_DATE])
@pytest.mark.parametrize("name", DOB_NAMES)
def test_a_date_of_birth_is_found_under_the_common_field_names(name: str, value: str) -> None:
    assert redacted_args({name: value}) == {name: "[DOB_1]"}
    assert redacted_json({"patient": {name: value}}) == {"patient": {name: "[DOB_1]"}}


ACCOUNT_NAMES = ["account", "accounts", "bank_accounts", "bankAccount", "accountNumber",
                 "accountNo", "account_no", "account_num", "AccountNumbers", "acct",
                 "accountnumber"]  # fmt: skip
NHS_NAMES = ["nhs", "NHS", "nhsNumber", "NHSNumber", "nhs_number", "nhs_no", "nhsnumber",
             "patient_nhs_number"]  # fmt: skip
SORT_NAMES = ["sort_code", "sortCode", "sortcode", "SortCode", "sort-code", "sortCodes"]


@pytest.mark.parametrize("name", ACCOUNT_NAMES)
def test_an_account_number_is_found_under_the_common_field_names(name: str) -> None:
    assert redacted_args({name: "12345678"}) == {name: "[ACCOUNT_1]"}


@pytest.mark.parametrize("name", NHS_NAMES)
def test_an_nhs_number_is_found_under_the_common_field_names(name: str) -> None:
    assert redacted_args({name: "943 476 5919"}) == {name: "[NHS_1]"}


@pytest.mark.parametrize("name", SORT_NAMES)
def test_a_sort_code_is_found_under_the_common_field_names(name: str) -> None:
    assert redacted_args({name: "20-45-67"}) == {name: "[SORT_CODE_1]"}


def test_field_names_that_only_look_similar_do_not_make_a_match() -> None:
    plain = {
        "account_type": "premium",
        "birth_place": "Leeds",
        "update_date": UK_DATE,
        "created": ISO_DATE,
        "birthplace_ref": "12345678",
        "dobson_ref": "12345678",
        "nhs_trust_code": "12345678",
    }
    assert redacted_args(plain) == plain
    assert redacted_json(plain) == plain


def test_a_list_under_a_field_name_takes_the_field_name_for_each_item() -> None:
    assert redacted_args({"account": ["12345678", "87654321"]}) == {
        "account": ["[ACCOUNT_1]", "[ACCOUNT_2]"]
    }
    assert redacted_args({"dob": [UK_DATE]}) == {"dob": ["[DOB_1]"]}
    assert redacted_json({"accounts": ["12345678"]}) == {"accounts": ["[ACCOUNT_1]"]}


# --- label and value, and the name of the tool ------------------------------------------------


@pytest.mark.parametrize("label_key", ["label", "name", "field", "key", "title"])
@pytest.mark.parametrize("order", ["label first", "value first"])
def test_a_value_takes_its_label_as_context(label_key: str, order: str) -> None:
    pair = (
        {label_key: "Date of birth", "value": UK_DATE}
        if order == "label first"
        else {"value": UK_DATE, label_key: "Date of birth"}
    )
    expected = {**pair, "value": "[DOB_1]"}
    assert redacted_args({"fields": [pair]}) == {"fields": [expected]}
    assert redacted_json([pair]) == [expected]


def test_a_value_with_a_neutral_label_is_unchanged() -> None:
    pair = {"label": "Reference", "value": UK_DATE}
    assert redacted_json([pair]) == [pair]


def tool_result_after(name: str, content: str) -> Any:
    call = AIMessage("", tool_calls=[{"name": name, "args": {}, "id": "c1"}])
    return redaction.prepare(
        REDACTOR, [HumanMessage("go"), call, ToolMessage(content, tool_call_id="c1")]
    ).messages[-1]


def test_a_bare_tool_result_takes_the_tool_name_as_context() -> None:
    assert tool_result_after("get_date_of_birth", UK_DATE).content == "[DOB_1]"
    assert tool_result_after("getAccountNumber", "12345678").content == "[ACCOUNT_1]"
    assert tool_result_after("lookup_nhs_number", "943 476 5919").content == "[NHS_1]"
    assert tool_result_after("get_reference", UK_DATE).content == UK_DATE
    named = sent(ToolMessage(UK_DATE, tool_call_id="x", name="get_date_of_birth"))
    assert named.content == "[DOB_1]"


def test_an_anthropic_tool_result_takes_the_name_of_the_tool_use_it_answers() -> None:
    use = AIMessage(
        content=[{"type": "tool_use", "id": "t1", "name": "get_date_of_birth", "input": {}}]
    )
    result = HumanMessage(
        content=[{"type": "tool_result", "tool_use_id": "t1", "content": UK_DATE}]
    )
    out = redaction.prepare(REDACTOR, [HumanMessage("go"), use, result]).messages[-1]
    assert isinstance(out.content, list)
    block = out.content[0]
    assert isinstance(block, dict)
    assert block["content"] == "[DOB_1]"


# --- untouched parts of JSON text stay exactly as they were ---------------------------------------


def test_json_text_is_changed_only_where_a_value_was_replaced() -> None:
    text = (
        '{ "rent": 1250.10, "big": 12345678901234567890, "a": 1, "a": 2,\n'
        '  "dob" : "12/03/1980", "x": 1e3 }'
    )
    expected = text.replace('"12/03/1980"', '"[DOB_1]"')
    assert sent(ToolMessage(text, tool_call_id="1")).content == expected
    # Nested JSON text inside a string is written back as a string, the rest untouched.
    inner = '{"dob":"12/03/1980","n":1.50}'
    outer = json.dumps({"payload": inner, "n": 2.0})
    out = sent(ToolMessage(outer, tool_call_id="1")).content
    assert out == json.dumps({"payload": '{"dob":"[DOB_1]","n":1.50}', "n": 2.0})


def test_json_nested_too_deep_to_read_is_refused_rather_than_sent() -> None:
    from app.llm.core.redaction import RedactionFailed

    deep = "[" * 5000 + "]" * 5000
    with pytest.raises(RedactionFailed):
        sent(ToolMessage(deep, tool_call_id="1"))


# --- JSON keys are text too -----------------------------------------------------------------------

EMAIL_KEY = "jo.bloggs@example.com"
PHONE_KEY = "07700 900123"
NAME_KEY = "Jo Bloggs"


def test_an_email_key_in_a_json_tool_result_is_replaced_and_the_rest_is_untouched() -> None:
    text = '{ "' + EMAIL_KEY + '" : "landlord",  "rent": 1250.10, "ok": true }'
    out = sent(ToolMessage(text, tool_call_id="1")).content
    assert out == text.replace(EMAIL_KEY, "[EMAIL_1]")


async def test_a_known_name_used_as_a_key_is_replaced_in_json_text_and_in_human_json() -> None:
    text = json.dumps({NAME_KEY: {"role": "tenant"}, "n": 1})
    async with redaction_scope(known={"NAME": [NAME_KEY]}):
        tool = sent(ToolMessage(text, tool_call_id="1")).content
        human = sent(HumanMessage(text)).content
    assert tool == human == text.replace(NAME_KEY, "[NAME_1]")


def test_keys_in_tool_call_arguments_are_replaced_parsed_and_as_a_raw_string() -> None:
    args = {PHONE_KEY: "mobile", "inner": {EMAIL_KEY: ["x"]}}
    assert redacted_args(args) == {"[PHONE_1]": "mobile", "inner": {"[EMAIL_1]": ["x"]}}
    raw = json.dumps(args)
    call = {"id": "1", "type": "function", "function": {"name": "n", "arguments": raw}}
    out = sent(AIMessage("", additional_kwargs={"tool_calls": [call]}))
    arguments = out.additional_kwargs["tool_calls"][0]["function"]["arguments"]
    assert PHONE_KEY not in arguments
    assert EMAIL_KEY not in arguments
    assert json.loads(arguments) == {"[PHONE_1]": "mobile", "inner": {"[EMAIL_1]": ["x"]}}


def test_an_anthropic_tool_use_input_and_a_tool_result_replace_keys() -> None:
    block = {"type": "tool_use", "id": "t1", "name": "n", "input": {EMAIL_KEY: "x"}}
    out = sent(AIMessage(content=[block]))
    assert isinstance(out.content, list)
    first = out.content[0]
    assert isinstance(first, dict)
    assert first["input"] == {"[EMAIL_1]": "x"}
    result = {"type": "tool_result", "tool_use_id": "t1", "content": json.dumps({EMAIL_KEY: 1})}
    out2 = sent(HumanMessage(content=[result]))
    assert isinstance(out2.content, list)
    second = out2.content[0]
    assert isinstance(second, dict)
    assert json.loads(second["content"]) == {"[EMAIL_1]": 1}


async def test_trace_masking_replaces_keys() -> None:
    from app.llm.langchain.redaction import masking_anonymizer

    masked = masking_anonymizer(REDACTOR)(
        {"inputs": {EMAIL_KEY: "x"}, "outputs": {"n": [PHONE_KEY]}}
    )
    assert masked == {"inputs": {"[EMAIL_1]": "x"}, "outputs": {"n": ["[PHONE_1]"]}}


async def test_the_tool_gets_the_original_keys_back() -> None:
    from langchain_core.language_models import GenericFakeChatModel

    from app.llm.langchain.redaction import RedactingChatModel

    answer = AIMessage(
        "",
        tool_calls=[
            {"name": "save", "args": {"[EMAIL_1]": "landlord", "n": {"[PHONE_1]": 1}}, "id": "1"}
        ],
    )
    model = RedactingChatModel(GenericFakeChatModel(messages=iter([answer])), REDACTOR)
    history = [
        HumanMessage("save it"),
        AIMessage(
            "", tool_calls=[{"name": "s", "args": {EMAIL_KEY: "x", PHONE_KEY: 1}, "id": "0"}]
        ),
    ]
    out = await model.ainvoke(history)
    assert out.tool_calls[0]["args"] == {EMAIL_KEY: "landlord", "n": {PHONE_KEY: 1}}


def test_keys_that_hold_nothing_personal_stay_byte_for_byte() -> None:
    text = '{ "rent" : 1250.10,\n "b": {"c": [1, 2]}, "rent": 2 }'
    assert sent(ToolMessage(text, tool_call_id="1")).content == text
    args = {"rent": 1, "nested": {"c": "x"}}
    assert redacted_args(args) == args


# --- the rewrite is linear ------------------------------------------------------------------------


def test_rewriting_json_text_with_many_changed_strings_is_fast() -> None:
    import time

    from app.llm.core.redaction_data import walk_text

    text = json.dumps([f"value {n} " + "x" * 70 for n in range(32_000)])  # about 3 MB
    started = time.perf_counter()
    out = walk_text(text, lambda value, _context: value.upper())
    took = time.perf_counter() - started
    assert json.loads(out)[31_999].startswith("VALUE 31999")
    assert took < 1.0, took  # the old rewrite, one slice per edit, took seconds


def test_company_accounts_before_a_number_is_a_known_false_positive() -> None:
    # Documented (#97): prose with "accounts" shortly before an 8-digit company number reads as an
    # account number. The model loses the reference; the user still sees it restored.
    text = "Companies House accounts filed, company No. 12345678"
    assert REDACTOR.redact([text]).texts == (
        "Companies House accounts filed, company No. [ACCOUNT_1]",
    )

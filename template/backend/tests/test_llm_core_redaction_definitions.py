"""The core's rules for tool definitions and output schemas (#78, #112's review): every string a
provider reads is scanned, the identifiers it checks are not, nothing changes when nothing matches,
and a Pydantic model is sent as a subclass with the redacted schema and comes back as the caller's
class. No framework is imported; the adapters' own tests show both frameworks use these rules.
The values are synthetic."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from app.llm.core.redaction import Redactor
from app.llm.core.redaction_data import (
    Restorer,
    walk_definition,
    walk_output_schema,
    with_schema,
)
from app.llm.redaction_rules import DETECTORS

EMAIL = "priya.shah@example.com"
REDACTOR = Redactor(DETECTORS)


def redact(value: Any) -> tuple[Any, dict[str, str]]:
    """`value` walked the way an adapter walks it: collect, redact in one call, replace."""
    strings: list[str] = []
    contexts: list[str] = []

    def collect(text: str, context: str) -> str:
        strings.append(text)
        contexts.append(context)
        return text

    walk_output_schema(value, collect)
    result = REDACTOR.redact(strings, contexts=contexts)
    replacements = iter(result.texts)
    out = walk_output_schema(value, lambda _text, _context: next(replacements))
    assert next(replacements, None) is None
    return out, dict(result.mapping)


def test_descriptions_titles_and_allowed_or_example_values_are_scanned() -> None:
    schema = {
        "title": f"Lead for {EMAIL}",
        "description": f"Write to {EMAIL}",
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": f"never {EMAIL}",
                "enum": [EMAIL, "office"],
                "default": EMAIL,
                "examples": [{"name": EMAIL}],
                "const": EMAIL,
            }
        },
    }
    out, mapping = redact(schema)
    assert mapping == {"[EMAIL_1]": EMAIL}
    assert EMAIL not in repr(out)
    to = out["properties"]["to"]
    assert to["enum"] == ["[EMAIL_1]", "office"]
    # Inside example data, even a `name` is the example's own text, not an identifier.
    assert to["examples"] == [{"name": "[EMAIL_1]"}]
    assert (to["default"], to["const"]) == ("[EMAIL_1]", "[EMAIL_1]")


def test_names_types_formats_and_references_are_identifiers_and_left_alone() -> None:
    # Every identifier, and the name a $ref points to, is the same string as one description: the
    # walker must hand that string to the redactor exactly once (the description), so a known
    # value equal to an identifier is still sent as it is there.
    tool = {
        "type": "Priya",
        "function": {
            "name": "Priya",
            "parameters": {
                "type": "Priya",
                "properties": {"x": {"$ref": "Priya", "format": "Priya", "$id": "Priya"}},
                "$defs": {"Priya": {"type": "Priya", "description": "Priya"}},
            },
        },
    }
    strings: list[str] = []

    def collect(text: str, _context: str) -> str:
        strings.append(text)
        return text

    walk_definition(tool, collect)
    assert strings.count("Priya") == 1
    # A replacement standing in for a known value reaches only the description.
    out = walk_definition(tool, lambda text, _context: "[NAME_1]" if text == "Priya" else text)
    assert out["type"] == out["function"]["name"] == "Priya"
    assert out["function"]["parameters"]["properties"]["x"] == {
        "$ref": "Priya",
        "format": "Priya",
        "$id": "Priya",
    }
    assert out["function"]["parameters"]["$defs"] == {
        "Priya": {"type": "Priya", "description": "[NAME_1]"}
    }


def test_a_property_name_and_the_required_entry_naming_it_get_the_same_placeholder() -> None:
    schema = {"type": "object", "properties": {EMAIL: {"type": "string"}}, "required": [EMAIL]}
    out, _ = redact(schema)
    assert out == {
        "type": "object",
        "properties": {"[EMAIL_1]": {"type": "string"}},
        "required": ["[EMAIL_1]"],
    }


def test_a_value_is_read_with_its_fields_name_in_front() -> None:
    # A bare date is only a date of birth with the field's name beside it.
    schema = {"properties": {"date_of_birth": {"type": "string", "enum": ["12/03/1980"]}}}
    out, _ = redact(schema)
    assert out["properties"]["date_of_birth"]["enum"] == ["[DOB_1]"]


def test_a_definition_with_nothing_to_replace_is_the_same_object() -> None:
    schema = {"type": "object", "description": "a freehold in Leeds"}
    out, mapping = redact(schema)
    assert out is schema
    assert mapping == {}


def test_anything_but_plain_json_data_is_refused() -> None:
    for bad in ({"description": object()}, {1: "x"}, {"enum": [object()]}, ["not a mapping"]):
        with pytest.raises(TypeError):
            walk_output_schema(bad, lambda text, _context: text)


class Contact(BaseModel):
    """Contact details. The office is priya.shah@example.com."""

    email: str = Field(description="The address", examples=["priya.shah@example.com"])


def test_a_model_is_sent_as_a_subclass_whose_schema_is_redacted() -> None:
    out, _ = redact(Contact)
    assert out is not Contact
    assert issubclass(out, Contact)
    assert out.__name__ == "Contact"  # OpenAI sends it as the schema's name
    sent = out.model_json_schema()
    assert sent["description"] == "Contact details. The office is [EMAIL_1]."
    assert sent["properties"]["email"]["examples"] == ["[EMAIL_1]"]
    assert EMAIL not in repr(sent)
    # The caller's class is unchanged.
    assert EMAIL in repr(Contact.model_json_schema())


def test_a_model_with_nothing_to_replace_is_the_same_class() -> None:
    class Plain(BaseModel):
        postcode: str

    out, _ = redact(Plain)
    assert out is Plain


def test_an_answer_parsed_into_the_subclass_comes_back_as_the_callers_class() -> None:
    redacted = with_schema(Contact, {"type": "object"})
    parsed = redacted.model_validate({"email": "[EMAIL_1]"})
    restored = Restorer({"[EMAIL_1]": EMAIL}).data(parsed)
    assert type(restored) is Contact
    assert restored.email == EMAIL
    assert restored.model_fields_set == {"email"}

"""An example eval: replace it with evals of the product's own prompts and agents."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import testing as t

from app.llm.gateway import Gateway

pytestmark = [pytest.mark.anyio, pytest.mark.eval]


@pytest.mark.langsmith
@pytest.mark.parametrize(
    ("question", "expected"), [("What is 12 x 12?", "144"), ("17 + 25?", "42")]
)
async def test_answers_arithmetic_with_the_number_alone(
    gateway: Gateway, question: str, expected: str
) -> None:
    t.log_inputs({"question": question})
    t.log_reference_outputs({"answer": expected})
    answer = await gateway.model().ainvoke(
        [SystemMessage("Answer with the number only."), HumanMessage(question)]
    )
    t.log_outputs({"answer": answer.text})
    t.log_feedback(key="exact", score=answer.text.strip() == expected)
    assert expected in answer.text

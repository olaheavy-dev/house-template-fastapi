"""The signed-in user: the smallest example of a protected, versioned route with a response
model. Replace or extend it once the product has a user table."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.auth import CurrentUser

router = APIRouter(prefix="/me", tags=["me"])


class MeOut(BaseModel):
    id: str


@router.get("")
async def me(user: CurrentUser) -> MeOut:
    return MeOut(id=user.id)

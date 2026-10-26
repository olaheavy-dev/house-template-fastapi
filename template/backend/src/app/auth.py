"""Who is calling: the hook every protected route and per-user limit hangs on.

The template can't know a project's identity provider (Supabase, Clerk, Auth0, its own sessions),
so `authenticate` is the one function a project replaces: read the request's credentials, verify
them, return a User, or None. Until it does, the app fails closed: in production every protected
route answers 401. Locally and in tests, an X-Dev-User header stands in for a signed-in user, so
the app can be built and tested before real sign-in exists; production ignores it.

    @router.get("/courses")
    async def my_courses(user: CurrentUser, session: SessionDep) -> list[CourseOut]:
        ...

User.id is an opaque, stable id (the provider's subject), never an email: it is what model calls
are tagged and budgeted by (Gateway.call_config) and what owned rows point at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.problems import Problem
from app.settings import Settings

DEV_USER_HEADER = "X-Dev-User"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:@-]{1,200}$")


@dataclass(frozen=True, slots=True)
class User:
    id: str


class NotAuthenticated(Problem):
    def __init__(self) -> None:
        super().__init__(401, detail="Sign in to use this.", headers={"WWW-Authenticate": "Bearer"})


async def authenticate(request: Request, settings: Settings) -> User | None:
    """Replace this with the project's real verification (a session cookie, a bearer token
    checked against the provider's keys). Return None when the request carries no valid
    credentials."""
    if settings.app_env == "production":
        return None
    claimed = request.headers.get(DEV_USER_HEADER, "")
    return User(id=claimed) if _SAFE_ID.match(claimed) else None


async def optional_user(request: Request) -> User | None:
    settings: Settings = request.app.state.settings
    return await authenticate(request, settings)


async def current_user(user: Annotated[User | None, Depends(optional_user)]) -> User:
    if user is None:
        raise NotAuthenticated
    return user


CurrentUser = Annotated[User, Depends(current_user)]
OptionalUser = Annotated[User | None, Depends(optional_user)]

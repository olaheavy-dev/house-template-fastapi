"""The product's API, versioned: every route the product adds joins `router` (mounted at /v1).

A breaking change goes to a new /v2 router beside this one, so existing clients keep working while
they move. Health checks and docs stay outside the version.

    # app/api/courses.py
    router = APIRouter(prefix="/courses", tags=["courses"])

    @router.get("")
    async def list_courses(
        user: CurrentUser, session: SessionDep, params: PageParamsDep
    ) -> Page[CourseOut]:
        after = params.after(int)  # the last id of the previous page, or None
        query = select(Course).where(Course.owner_id == user.id).order_by(Course.id)
        if after is not None:
            query = query.where(Course.id > after)
        rows = (await session.scalars(query.limit(params.limit + 1))).all()
        return page(rows, params, key=lambda c: c.id, item=CourseOut.model_validate)

    # here: router.include_router(courses.router)
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api import me

router = APIRouter(prefix="/v1")
router.include_router(me.router)

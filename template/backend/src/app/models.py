"""The product's tables. Import every model module here: Alembic's autogenerate reads Base.metadata
after importing this module, so a model it can't see gets no migration.

    class Course(Base):
        __tablename__ = "course"
        id: Mapped[int] = mapped_column(primary_key=True)
        title: Mapped[str]
        embedding: Mapped[list[float] | None] = mapped_column(VECTOR(1536))

        __table_args__ = (
            Index("ix_course_embedding", embedding, postgresql_using="hnsw",
                  postgresql_with={"m": 16, "ef_construction": 64},
                  postgresql_ops={"embedding": "vector_cosine_ops"}),
        )
"""

from __future__ import annotations

from app.db import Base
from app.llm.budget import LlmCall, LlmSpendDay, LlmSpendUserDay

__all__ = ["Base", "LlmCall", "LlmSpendDay", "LlmSpendUserDay"]

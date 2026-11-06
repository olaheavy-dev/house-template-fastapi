"""Imported before conftest.py and any app code: the tests never read backend/.env (a developer's
real keys and settings), only what each test sets."""

import os
from pathlib import Path

os.environ["ENV_FILE"] = ""

# backend/, where pyproject.toml, migrations/ and the compose files are.
BACKEND = Path(__file__).resolve().parents[1]

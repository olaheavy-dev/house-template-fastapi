"""Imported before conftest.py and any app code: the tests never read backend/.env (a developer's
real keys and settings), only what each test sets."""

import os

os.environ["ENV_FILE"] = ""

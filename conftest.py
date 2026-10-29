"""Pytest bootstrap.

Forces the project root onto sys.path so that `from kosh.resilience import ...`
resolves no matter where pytest is invoked from, on any platform, without
PYTHONPATH gymnastics.

Why this exists: by default pytest inserts the test file's directory (tests/)
onto sys.path, not the project root. That makes `import kosh` fail with
ModuleNotFoundError. This file is the belt; `pythonpath = .` in pytest.ini
and tests/__init__.py are the braces.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)
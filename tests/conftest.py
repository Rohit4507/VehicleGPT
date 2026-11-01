"""Make the repository root importable regardless of how pytest is invoked.

``python -m pytest`` puts the cwd on ``sys.path``; a bare ``pytest`` does not.
Without this, ``import sim`` and ``import vehiclegpt`` only work in some
invocations - a difference that shows up in CI and nowhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

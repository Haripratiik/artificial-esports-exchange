"""The browser front end for the simulator.

Puts the simulator's own package directory on the import path, because the
project is laid out with ``arena`` under ``python/`` and is not installed. Under
pytest that is handled by ``pythonpath`` in ``pyproject.toml``, which is exactly
why it went unnoticed: every test passed while ``python -m dashboard.server``
(the command in the module docstring, and the only way anyone actually opens
this thing) died on ``ModuleNotFoundError: No module named 'arena'``.

Doing it here rather than in ``server.py`` covers every entry into the package,
including importing ``dashboard.state`` directly from a REPL or a script.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python"

if _PACKAGE_ROOT.is_dir():
    path = str(_PACKAGE_ROOT)
    if path not in sys.path:
        # Appended rather than prepended: an installed copy of ``arena`` should
        # win over the working tree, which is the behaviour anyone who has
        # actually installed the package will expect.
        sys.path.append(path)

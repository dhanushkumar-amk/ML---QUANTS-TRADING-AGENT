# ============================================================
# Pytest Global Configuration & Fixtures
# ============================================================
"""Global fixtures and headless environment setup for CI/CD test runs."""

from __future__ import annotations

import sys
import types

# Ensure headless Agg backend for matplotlib across all test runners
# Handle Windows Application Control policy fallback if _c_internal_utils is restricted
if "matplotlib._c_internal_utils" not in sys.modules:
    try:
        import matplotlib._c_internal_utils  # noqa: F401
    except ImportError:
        sys.modules["matplotlib._c_internal_utils"] = types.ModuleType(
            "matplotlib._c_internal_utils"
        )

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

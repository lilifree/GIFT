"""Install GIFT in the driver and in every Python worker process."""

from __future__ import annotations

import os


if os.environ.get("GIFT_ENABLE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from gift_runtime.patch import apply_runtime_patches

    apply_runtime_patches()

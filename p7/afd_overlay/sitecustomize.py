"""Imported by every Python process whose PYTHONPATH contains /opt/afd."""

import os

if os.environ.get("AFD_ENABLE") == "1":
    from afd_overlay import install

    install()

"""Minimal FinRL utilities required for the FinRL-DT replication."""
from __future__ import annotations

import os
from typing import Iterable


def check_and_make_directories(directories: Iterable[str]) -> None:
    """Create directories relative to the repository root if they are missing."""
    for directory in directories:
        path = os.path.join("./", directory)
        os.makedirs(path, exist_ok=True)

"""Filesystem roots. Data lives outside the repo; override with DQITT_DATA."""

import os
from pathlib import Path

DATA = Path(os.environ.get("DQITT_DATA", "/home/al/Data"))

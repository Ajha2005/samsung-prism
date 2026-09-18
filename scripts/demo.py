#!/usr/bin/env python3
"""Retrieval demo on real queries. Thin wrapper around ``prism.cli demo``.

    python scripts/demo.py --backend hashing
    python scripts/demo.py --backend hashing --versioned
"""

import sys

from prism.cli import demo_main

if __name__ == "__main__":
    raise SystemExit(demo_main(sys.argv[1:]))

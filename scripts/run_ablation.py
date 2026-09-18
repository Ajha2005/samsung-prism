#!/usr/bin/env python3
"""Ablation sweep. Thin wrapper around ``prism.cli ablation``.

    python scripts/run_ablation.py --backend hashing          # offline synthetic set
    python scripts/run_ablation.py --apps                     # real AppsRetrieval
"""

import sys

from prism.cli import ablation_main

if __name__ == "__main__":
    raise SystemExit(ablation_main(sys.argv[1:]))

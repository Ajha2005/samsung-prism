#!/usr/bin/env python3
"""Run the CoIR AppsRetrieval MTEB eval and write the submission JSON.

Thin wrapper around ``prism.cli eval``. Example:

    python scripts/run_eval.py --config config/e5_base.json \
        --output results/appsretrieval_results.json
"""

import sys

from prism.cli import eval_main

if __name__ == "__main__":
    raise SystemExit(eval_main(sys.argv[1:]))

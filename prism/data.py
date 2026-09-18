"""Small synthetic datasets that mirror the CoIR AppsRetrieval shape.

Each item is a natural-language query paired with the Python snippet that
answers it — the same query->code retrieval task the leaderboard scores, just
tiny and local so the demo, offline eval, and tests run with no network. This is
a fixture for exercising the pipeline, not a substitute for the real dataset.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# (doc_id, query, code) triples. The query is what a user would type; the code is
# the relevant snippet. Queries describe behavior rather than name the function,
# so lexical overlap alone is not enough — the retrieval has to work.
_ITEMS: List[Tuple[str, str, str]] = [
    (
        "two_sum",
        "find two numbers in a list that add up to a target",
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        if target - x in seen:\n"
        "            return [seen[target - x], i]\n"
        "        seen[x] = i\n"
        "    return []",
    ),
    (
        "max_subarray",
        "largest sum of any contiguous subarray",
        "def max_subarray(a):\n"
        "    best = cur = a[0]\n"
        "    for x in a[1:]:\n"
        "        cur = max(x, cur + x)\n"
        "        best = max(best, cur)\n"
        "    return best",
    ),
    (
        "binary_search",
        "search for a value in a sorted array",
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        if arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1",
    ),
    (
        "dfs",
        "traverse a graph in depth first order",
        "def dfs(graph, start):\n"
        "    stack, seen = [start], set()\n"
        "    while stack:\n"
        "        node = stack.pop()\n"
        "        if node in seen:\n"
        "            continue\n"
        "        seen.add(node)\n"
        "        stack.extend(graph[node])\n"
        "    return seen",
    ),
    (
        "bfs",
        "shortest path length between two nodes in an unweighted graph",
        "from collections import deque\n\n"
        "def bfs_dist(graph, src, dst):\n"
        "    q = deque([(src, 0)])\n"
        "    seen = {src}\n"
        "    while q:\n"
        "        node, d = q.popleft()\n"
        "        if node == dst:\n"
        "            return d\n"
        "        for nxt in graph[node]:\n"
        "            if nxt not in seen:\n"
        "                seen.add(nxt)\n"
        "                q.append((nxt, d + 1))\n"
        "    return -1",
    ),
    (
        "merge_intervals",
        "combine overlapping intervals into merged ranges",
        "def merge_intervals(intervals):\n"
        "    intervals.sort()\n"
        "    out = [intervals[0]]\n"
        "    for s, e in intervals[1:]:\n"
        "        if s <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n"
        "            out.append([s, e])\n"
        "    return out",
    ),
    (
        "gcd",
        "greatest common divisor of two integers",
        "def gcd(a, b):\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a",
    ),
    (
        "sieve",
        "list all prime numbers up to n",
        "def primes_up_to(n):\n"
        "    sieve = [True] * (n + 1)\n"
        "    sieve[0:2] = [False, False]\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if sieve[i]:\n"
        "            for j in range(i * i, n + 1, i):\n"
        "                sieve[j] = False\n"
        "    return [i for i, p in enumerate(sieve) if p]",
    ),
    (
        "quicksort",
        "sort a list using the quicksort algorithm",
        "def quicksort(a):\n"
        "    if len(a) <= 1:\n"
        "        return a\n"
        "    pivot = a[len(a) // 2]\n"
        "    left = [x for x in a if x < pivot]\n"
        "    mid = [x for x in a if x == pivot]\n"
        "    right = [x for x in a if x > pivot]\n"
        "    return quicksort(left) + mid + quicksort(right)",
    ),
    (
        "is_palindrome",
        "check whether a string reads the same backwards",
        "def is_palindrome(s):\n"
        "    return s == s[::-1]",
    ),
    (
        "word_count",
        "count how many times each word appears in a text",
        "from collections import Counter\n\n"
        "def word_count(text):\n"
        "    return Counter(text.lower().split())",
    ),
    (
        "fibonacci",
        "generate the nth fibonacci number iteratively",
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a",
    ),
    (
        "matrix_transpose",
        "transpose a two dimensional matrix",
        "def transpose(matrix):\n"
        "    return [list(row) for row in zip(*matrix)]",
    ),
    (
        "levenshtein",
        "edit distance between two strings",
        "def edit_distance(a, b):\n"
        "    dp = list(range(len(b) + 1))\n"
        "    for i, ca in enumerate(a, 1):\n"
        "        prev, dp[0] = dp[0], i\n"
        "        for j, cb in enumerate(b, 1):\n"
        "            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))\n"
        "    return dp[-1]",
    ),
    (
        "flatten",
        "flatten a nested list into a single list",
        "def flatten(nested):\n"
        "    out = []\n"
        "    for item in nested:\n"
        "        if isinstance(item, list):\n"
        "            out.extend(flatten(item))\n"
        "        else:\n"
        "            out.append(item)\n"
        "    return out",
    ),
    (
        "lru_cache_manual",
        "cache the most recently used items with a size limit",
        "from collections import OrderedDict\n\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity):\n"
        "        self.cap = capacity\n"
        "        self.store = OrderedDict()\n"
        "    def get(self, key):\n"
        "        if key not in self.store:\n"
        "            return -1\n"
        "        self.store.move_to_end(key)\n"
        "        return self.store[key]\n"
        "    def put(self, key, value):\n"
        "        self.store[key] = value\n"
        "        self.store.move_to_end(key)\n"
        "        if len(self.store) > self.cap:\n"
        "            self.store.popitem(last=False)",
    ),
    (
        "roman",
        "convert an integer to a roman numeral",
        "def to_roman(n):\n"
        "    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),\n"
        "            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n"
        "    out = []\n"
        "    for v, sym in vals:\n"
        "        while n >= v:\n"
        "            out.append(sym); n -= v\n"
        "    return ''.join(out)",
    ),
    (
        "anagram_groups",
        "group words that are anagrams of each other",
        "from collections import defaultdict\n\n"
        "def group_anagrams(words):\n"
        "    groups = defaultdict(list)\n"
        "    for w in words:\n"
        "        groups[''.join(sorted(w))].append(w)\n"
        "    return list(groups.values())",
    ),
]


def load_synthetic_corpus() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Return (corpus, queries, qrels) for the synthetic AppsRetrieval-like set."""
    corpus = {doc_id: code for doc_id, _, code in _ITEMS}
    queries = {f"q_{doc_id}": query for doc_id, query, _ in _ITEMS}
    qrels = {f"q_{doc_id}": {doc_id: 1} for doc_id, _, _ in _ITEMS}
    return corpus, queries, qrels


def load_versioned_corpus() -> Dict[str, List[str]]:
    """Return {group_id: [version_text, ...]} for the cross-version demo.

    Each group holds near-identical variants of one function that differ by a
    distinctive behavioral change — the case the evolutionary representation is
    meant to keep distinguishable.
    """
    return {
        "sort": [
            "def sort_items(items):\n    return sorted(items)",
            "def sort_items(items):\n    return sorted(items, reverse=True)",
            "def sort_items(items):\n    return sorted(items, key=lambda x: abs(x))",
        ],
        "read_file": [
            "def read_file(path):\n    with open(path) as f:\n        return f.read()",
            "def read_file(path):\n    with open(path, encoding='utf-8') as f:\n        return f.read()",
            "def read_file(path):\n    with open(path, 'rb') as f:\n        return f.read()",
        ],
        "dedupe": [
            "def dedupe(xs):\n    return list(set(xs))",
            "def dedupe(xs):\n    seen = set()\n    return [x for x in xs if not (x in seen or seen.add(x))]",
        ],
    }


# A distinctive query per (group, version index): what someone would type when
# they want *that specific* version, not just the family.
_VERSIONED_QUERIES: Dict[str, List[str]] = {
    "sort": [
        "sort items in ascending order",
        "sort items in reverse descending order",
        "sort items by absolute value magnitude",
    ],
    "read_file": [
        "read the contents of a text file",
        "read a file decoded as utf-8 text",
        "read a file as raw binary bytes",
    ],
    "dedupe": [
        "remove duplicate elements from a list",
        "remove duplicates while preserving original order",
    ],
}


def load_versioned_eval() -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Return (groups, queries, qrels) for cross-version retrieval evaluation.

    Corpus doc ids are ``"{group}@v{i}"``; each distinctive query is relevant to
    exactly one version, so a metric can tell whether the *right* version was
    retrieved among its near-identical siblings.
    """
    groups = load_versioned_corpus()
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    for gid, versions in groups.items():
        q_texts = _VERSIONED_QUERIES.get(gid, [])
        for i in range(len(versions)):
            if i >= len(q_texts):
                continue
            qid = f"q_{gid}_v{i}"
            queries[qid] = q_texts[i]
            qrels[qid] = {f"{gid}@v{i}": 1}
    return groups, queries, qrels

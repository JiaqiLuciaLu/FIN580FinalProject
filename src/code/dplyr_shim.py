"""Python shims for R `dplyr` functions used across the portfolio-creation
modules. Each R file does `library('dplyr')`; each Python translation
imports from here.
"""

import numpy as np


def ntile(x, q_num):
    """Translation of dplyr::ntile(x, q_num).

    Assigns 1-indexed bin labels 1..q_num based on rank order; bucket sizes
    differ by at most 1 (smaller-rank buckets can be one larger). Ties
    broken by order of occurrence (row_number, not rank).
    """
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(x, kind="stable")
    r = np.empty(n, dtype=np.int64)
    r[order] = np.arange(1, n + 1)
    return np.floor(q_num * (r - 1) / n).astype(int) + 1

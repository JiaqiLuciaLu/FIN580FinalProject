"""
Long-only AP-Pruning wrapper.

Reuses the full grid-search / CV / CSV-writing framework in
`src.code.ap_pruning.pruning` by scoping a monkey-patch on its module-level
`lasso_en` reference. This keeps the baseline pipeline and its R-validated
outputs untouched, while letting the ablation runner reuse every downstream
step (`pick_best_lambda`, `pick_sr_n`, `sdf_regression`) with no code changes.

Safety note on multiprocessing:
- `pruning._grid_search` creates a `Pool` with the OS default start method.
- On Linux that's fork; worker children inherit the patched attribute via
  copy-on-write memory at fork time.
- The context manager restores the original `lasso_en` reference on exit.
  Do NOT call `pruning.ap_pruning` and `ap_pruning_long_only` concurrently
  from the same process — the patch is process-global while the context is
  active.
"""

import contextlib

from src.code.ap_pruning import pruning
from .long_only_lasso import lasso_en_positive


@contextlib.contextmanager
def _patched_lasso(lasso_fn):
    original = pruning.lasso_en
    pruning.lasso_en = lasso_fn
    try:
        yield
    finally:
        pruning.lasso_en = original


def ap_pruning_long_only(*args, **kwargs):
    """
    Long-only analogue of `pruning.ap_pruning`. Identical signature — the
    only difference is that every LARS-EN solve enforces β_i ≥ 0.

    Writes the same `results_{cv_name}_l0_i_l2_j.csv` layout into `output_dir`,
    so all downstream tooling (`pick_best_lambda`, `pick_sr_n`, SR_N.csv,
    `sdf_regression`) works unchanged on the ablation output.
    """
    with _patched_lasso(lasso_en_positive):
        pruning.ap_pruning(*args, **kwargs)

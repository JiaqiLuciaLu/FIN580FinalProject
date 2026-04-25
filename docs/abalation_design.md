## Design

**Goal.** Replace the hard median split (`ntile(x, 2)`) in the AP-Tree builder with a soft/fuzzy split. Only the split step changes; downstream (step3 RmRF, step4 monoculture filter, LARS-EN pruning, SR / SDF / XS-R² metrics) consumes the same `{file_id}ret.csv` schema and runs unchanged.

1. **Node representation.** Each node carries a weight vector `w ∈ R^n` over the *full* month-*t* cross-section, with `w_i ≥ 0` the membership mass of firm *i*. Root: `w_i = 1` for every firm. Every firm appears in every node, usually with very small weight in most leaves.

2. **Within-parent median centering.** At every parent node, recompute `m = weighted_median(x, w^parent)` over *that node's own* `(x, w^parent)` — not a global or root-level median. Zero-weight firms are dropped before computing `m` (a firm with `w_i = 0` has fully taken the other branch upstream and must not shift the threshold). Centering the sigmoid at this within-parent median is what enforces sibling mass balance `Σ w^high ≈ Σ w^low ≈ (parent mass)/2` at every depth — matching the hard-split 50/50 behavior — and what makes the `α → ∞` limit recover the exact hard `ntile` split. Implementation: `fuzzy_tree_split.weighted_median` (`fuzzy_tree_split.py:36-72`), called per node from `fuzzy_split` (`fuzzy_tree_split.py:116-117`).

3. **Fuzzy split rule.** Given parent values `x_i ∈ [0, 1]` (quantile-normalized in step 1), parent weights `w_i^parent`, and the within-parent median `m` from §2:
   - `σ_i = σ(α · (x_i − m))` with `σ(u) = 1/(1+e^{−u})` and steepness `α > 0`
   - `w_i^high = σ_i · w_i^parent`
   - `w_i^low  = w_i^parent − w_i^high`  (subtraction, not `(1−σ)·w`, so conservation `w^high + w^low = w^parent` holds bit-exactly)

   v1 uses a single global `α = 10.0` (default in `step2`); `α` is the tuning hyperparameter in the ablation sweep.

4. **Why sigmoid, not softmax.** For a 2-way split the two are algebraically identical — a softmax `exp(α x_i) / (exp(α x_i) + exp(α m))` reduces to `σ(α(x_i − m))`. The substantive distinction is *per-firm* sigmoid (independent across firms, what we use) vs *cross-firm* softmax `p_i = exp(α x_i) / Σ_j exp(α x_j)`. A cross-firm softmax would couple every firm's high-side mass to the full parent distribution, break per-firm conservation `w^high + w^low = w^parent`, and in the `α → ∞` limit concentrate all mass on the *single top firm* instead of the *top half* — i.e. it would not reduce to the hard `ntile` split. The per-firm sigmoid keeps each firm's decision a local "above or below `m`" question and recovers `ntile` exactly in the limit.

5. **Tree build (`fuzzy_tree_month`).** BFS over levels `0..tree_depth`: at each level `i`, every parent vector is split using `feat_list[i]`, with children appended `(low, high)` to match `ntile`'s `k=1` low / `k=2` high convention. Output `ret_table` row shape `(2**(tree_depth+1) − 1,)` in level-major / low-then-high BFS order, identical to the hard-split helper, so column codes (`1, 11, 12, 111, …`) align across pipelines. v1 emits `{file_id}ret.csv` only — no per-leaf `feat_min` / `feat_max` (every firm sits in every leaf, so those collapse to cross-section bounds; deferred to a weighted-quantile follow-up if step4 ever needs them).

6. **Leaf return (with optional dead-stock pruning).** `r_L = Σ_i (w_i · size_i · r_i) / Σ_i (w_i · size_i)` — value-weight *and* fuzzy-weight, renormalized by `Σ w·size`. Optional `dead_threshold`: firms with `w_i < dead_threshold` are zeroed before averaging, and the denominator rescales the survivors to sum-to-1, dropping the long thin tail rather than just downweighting it. Used in the overfitting diagnostic at large `α` where most firms carry near-zero leaf mass.
 

## Example

Parent node has 4 firms A, B, C, D (ordered from largest to smallest size). Split characteristic is size. Quantile-normalized size values in the parent are `x = (0.95, 0.70, 0.30, 0.05)`; the within-parent median is `m = 0.5`. All parent weights are 1. Steepness `α = 10`.

- **Original hard-median method.** LEFT (small) = `[C, D]` (weights `[1, 1]`); RIGHT (large) = `[A, B]` (weights `[1, 1]`).
- **Fuzzy method** — `σ_i = σ(10 · (x_i − 0.5))`:
  - A: `σ(10·0.45) ≈ 0.989`
  - B: `σ(10·0.20) ≈ 0.881`
  - C: `σ(10·−0.20) ≈ 0.119`
  - D: `σ(10·−0.45) ≈ 0.011`

  - RIGHT (large, top): `[A, B, C, D]` with `w^high ≈ [0.989, 0.881, 0.119, 0.011]`, total mass `≈ 2.00`.
  - LEFT (small, bottom): `[A, B, C, D]` with `w^low ≈ [0.011, 0.119, 0.881, 0.989]`, total mass `≈ 2.00`.
  - Per-firm conservation: `w^high + w^low = 1 = w^parent` for every firm. ✓
  - Sibling mass: both children ≈ 2 (half the parent mass of 4). ✓
  - Limits: `α → ∞` ⇒ RIGHT = `[1, 1, 0, 0]`, LEFT = `[0, 0, 1, 1]` (hard median split recovered exactly). `α → 0` ⇒ all weights = 0.5 (degenerate).


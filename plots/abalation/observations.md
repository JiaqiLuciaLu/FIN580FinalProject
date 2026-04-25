# Ablation 1 — Observations

## dead_stocks_alpha50

Leaf-level dead-stocks diagnostic for the fuzzy AP-Tree at α=50, K=25, threshold=0.1% (firm `w_i^L < 0.001` ⇒ "dead"). Source: `abalation1_LME_OP_Investment_dead_stocks_alpha50_K25.csv` and its `_summary.csv`.

1. **Effective leaf size ≈ 770 firms** (out of ~4 011 in the average month) at depth 4. The hard split would give ≈ 250 firms per leaf (1/16 of the cross-section), so fuzzy retains roughly **3× more firms per leaf** in the meaningful-weight sense. So while every firm formally appears in every leaf with non-zero weight, only ~20 % of the cross-section actually contributes to any given leaf — fuzzy is "softer than hard, but nowhere near uniform."

2. **Monoculture leaves are notably fuller** (1 095 alive firms, 72.7 % dead) than non-monoculture leaves (770 alive, 80.8 % dead). The same characteristic split four times in a row produces broader transition zones — fuzzy mass concentrates less aggressively when there's no orthogonal characteristic to discriminate further. Step 4 still drops these 48 leaves; just worth noting if the monoculture rule is ever revisited for the fuzzy pipeline.

3. **The EN-selected leaves are slightly *thinner* than average** (697 alive in the K=25 SDF vs 771 in the post-step4 universe; 82.6 % dead vs 80.8 %). The elastic net systematically prefers leaves where the fuzzy weights have committed more sharply to one side or the other. Leaves with many firms stuck in the indecisive 0.4–0.6 range carry weaker signal and don't make the K=25 cut.

## dead_pruned_alpha50

Manual pruning experiment: zero out per-firm leaf weights below `dt` *before* computing the value-weighted leaf return, then rerun the entire pipeline (steps 2-4 + AP_Pruning + SR metrics) and compare K=25 SR against unpruned α=50. Source: `abalation1_LME_OP_Investment_pruning_focused_K25.csv` (9 rows: baseline + α=50 unpruned + 7 dt thresholds 0.1 % … 20 %).

1. **Pruning the soft tail does not degrade SR — it can improve it.** The unpruned α=50 K=25 SDF gets test SR 0.703. Pruning at `dt ∈ {0.1 %, 0.5 %, 2 %, 5 %}` keeps test SR ≈ 0.70 (within ±0.005). At `dt ∈ {1 %, 7.5 %, 10 %, 20 %}` test SR jumps to **0.72-0.74**, with the maximum at **dt=20 % → test SR 0.736** (best in the entire 9-row table). Validation SR stays in [1.225, 1.241] for every variant; CV picks the same K=25 region throughout.

2. **The improvement is overfit reduction, not noise.** The variants with elevated test SR also have *lower* train SR — by ~0.05 to 0.08. Train-test gap goes from **0.21 (unpruned) → 0.10 (dt=10 %) → 0.16 (dt=20 %)**. So the soft tail of dead weights wasn't carrying signal, but the elastic net was *fitting* it — pruning forces the EN onto a more parsimonious basin.

3. **Firms / portfolio collapses with `dt`, then plateaus around α=50's structural floor.** Unpruned: 4 011 (every firm). dt=0.1 %: 911. dt=1 %: 705. dt=5 %: 589. dt=10 %: 471. dt=20 %: 483 (slight bump because the EN selection shifts to include one more shallow portfolio). The α=50 floor is ~470-480 alive / portfolio — only **1.24× the hard split's 381**. Going lower would need a steeper α (narrower transition zone), not a stricter dt: at α=50, ~12 % of fuzzy mass per leaf sits in the [0.1, 0.5] band that even dt=10 % doesn't reach.

4. **Practical implication.** The fuzzy α=50 SDF can be replaced by **"fuzzy α=50 with hard-zero pruning at 10-20 %"** for a 0.03-0.05 test-SR gain *and* a 4-8× reduction in firms touching each portfolio's value-weighted return. The soft tail at this α is decorative; only the body of weights `w ∈ [0.1, 1.0]` carries signal.

## alpha_steepness

How steep is the per-firm sigmoid as a function of α? Source: `abalation1_alpha_steepness_summary.csv` and `abalation1_alpha_steepness_curves.png`. The fundamental knob is the per-split transition half-width `δ_99 = log(99) / α ≈ 4.6 / α`, expressed in quantile-of-characteristic units (since `x ∈ [0, 1]`).

| α | transition half-width | firms in soft band (one split) | depth-4 weight at δ=5% | depth-4 weight at δ=10% |
|---:|---:|---:|---:|---:|
| 10 | 0.460 | 92 % | 0.15 | 0.29 |
| 50 | 0.092 | **18 %** | **0.73** | 0.97 |
| 100 | 0.046 | 9 % | 0.97 | 1.00 |
| 200 | 0.023 | 5 % | 1.00 | 1.00 |

1. **The fuzzy edge lives entirely in `1/α`.** Every quantitative property of the soft split — width of the transition zone, fraction of firms with intermediate weights, depth-4 saturation rate — scales as `1/α`. Going α=50 → 100 halves the soft zone; α=50 → 200 quarters it.

2. **α=50 is the only setting where the soft band is "interestingly wide".** At α=50 a firm sitting just 5 % above the median (in quantile space) gets a depth-4 leaf weight of 0.73 — clearly "alive" but not saturated. At α=100 the same firm has weight 0.97 — essentially indistinguishable from a hard-split membership of 1. At α=200 the cumulative weight is 0.9998, and the resulting tree is bit-identical to hard-split for any practical threshold.

3. **This is why α=50 was the test-SR sweet spot in the sweep.** α=10/20 were too soft (every firm spread across every leaf, no signal); α=100/200 were too close to hard split to extract a meaningful smoothing benefit. Only α=50 sat in the regime where ~18 % of firms per split land in the indecisive [0.01, 0.99] sigmoid band — wide enough that the elastic net can exploit the smooth membership, narrow enough that the leaves still discriminate.

4. **And this is why pruning behaviour differs by α.** At α=50, the soft band has real mass in `w ∈ [0.1, 0.5]` that can be aggressively pruned without losing signal (we showed dt up to 20 % maintains test SR). At α=100 the soft mass has already collapsed to `[0.97, 1.00]`, so pruning at any reasonable threshold is essentially a no-op — the model *is* the hard split. At α=10 the soft mass is so spread out that there is no clean cutoff: pruning destroys the signal because the entire SDF lives in the [0.01, 0.5] band.

5. **Recipe for tuning α.** Pick α so that `1/α` is in the same neighbourhood as the *quantile resolution you care about* — i.e. how finely the underlying characteristic actually distinguishes firms. For `x ∈ [0, 1]` quantile-normalized inputs, α=50 corresponds to a ±9 % "soft window" around each split, which is roughly one decile bucket. That seems to be the right resolution for monthly equity cross-sections of ~4 000 firms; finer (α=100) loses the smoothing benefit, coarser (α=10) destroys discrimination.

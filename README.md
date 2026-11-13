# Top-Level Operational Equilibrium (TLOE): reproduction scripts

Scripts behind every number in the paper *Top-Level Operational Equilibrium* (V. Casamayor Pujol, N.-M. Rouska, B. Sedlak).
[[arXiv link and Zenodo DOI to be added]]

Requirements: Python 3.10+, numpy, scipy, matplotlib (figure only): `pip install -r requirements.txt`.
Run everything with `./run_all.sh` (exact checks: a few minutes; finite-sample study: about 1–2 h of single-core time).
Expected outputs are in `results/`.

## Models

- **J**: two-stage Jackson tandem, λ = 1, μ₁ = 2, μ₂ = 1.6 (truncated at 70 jobs per stage).
- **B** (B3) / **B2**: the same with a buffer of 3 / 2 at stage 2, blocking after service.
- **K-stage tandems** (`kstage.py`): truncated at a total of 11–24 jobs.
- **A**: queue with a hysteretic replica autoscaler (λ = 1.2, μ = 1 per replica, r ∈ {1, 2}, up at n ≥ 8, down at n ≤ 1, rate α).

Exact values: sparse linear solves and matrix-exponential actions; derivatives by central differences (h = 1e-5). Simulations: exact uniformisation from a stationary initial state.

## exact/

| Script | Paper | What it computes |
|---|---|---|
| `verify_tloe.py` | §5, Tables 1, 3 | product form; tests of (F) and (R); FDT predictions with N, n₁, n₂, −n₁ and the exact conjugate; reciprocity |
| `verify_independence_and_kl.py` | §4.3, §5.3, §6.5 | three-time symmetry (J) with positive control; stage-permutation invariance; (R) without (F); small-δ KL law; truncation |
| `kl_bridge.py` | §6.5 | KL between observed and tilted laws of N |
| `identifiability.py` | §6.2 | FDT with the top-level conjugate B^Y: static exact, transient not |
| `eigen_check.py` | §5.3 | generalised eigenvalues of the lagged covariances stay real in model B |
| `extra_checks.py` | §2.1, §4.1, §5, §6.5, §8.2 | capacity rows of Table 1; three-time asymmetry for B; Chapman–Kolmogorov test and implied time scales of N; decay time scale; KL extrapolation; error conventions |
| `weber_conditional_check.py`, `positive_control_erlang.py` | §5.2 | fixed-arrival interchangeability (exact) and its Erlang-2 positive control |
| `kstage.py`, `stage_controls.py` | §5.2, Table 2 | Lemma 2 (block interchangeability) and Corollary 2 (reversibility of (N, n_j)) |
| `autoscaler.py`, `autoscaler_table.py`, `autoscaler_sufficiency.py`, `compare_metrics.py` | §5.4, Table 4 | model A across α; minimal sufficient statistic; same metrics for J, B, B2 |
| `autoscaler_lumped_db.py` | §5.4 | model A: lumped detailed balance fails on the level sets of the score, so no sufficient aggregate is reversible [numerical] |
| `cross_scale_chain_rule.py` | §8.2 | cross-scale rule on a modulated M/M/1 queue |
| `ps_vs_fcfs_erlang.py` | §10 | single queue with Erlang-2 service: n is sufficient for the load and the prediction is exact under processor sharing, not under FCFS [numerical] |
| `blocking_minimal_statistic.py` | §5.3 | model B: the minimal sufficient statistic for the load is the full state (no admissible aggregate); bias of the prediction with B = N against buffer size [numerical] |
| `fig_evidence.py` | Fig. 1 | evidence figure (PDF and PNG) |
| `exact.py` | — | shared tandem builder |

## finite_sample/

| Script | Paper | What it computes |
|---|---|---|
| `exact.py`, `sim.py`, `models.py`, `autoscaler.py` | §7 | exact references and exact simulators for J, B3, B2, A |
| `study.py MODEL [NCHUNK] [SENS]` | §7 | prediction error, T-R test and baseline cross-moment test, per replicate; sensitivity runs; writes `results/study_MODEL.json` |
| `summarise.py` | Tables 5, sensitivity | medians/p90 with bootstrap CIs, rejection rates with Wilson CIs |
| `tf_power.py`, `tf_power_A.py` | §7.4, Table 6 | cost of a dynamic T-F test with hypothesised and estimated conjugates |
| `tr_signal_strength.py` | §7 | exact size of the time-asymmetry signal |

## LLM statement
The authors used Claude to assist with writing the verification and simulation scripts. The authors reviewed and verified all content and take full responsibility for it.

### License: MIT




# CACS hyperparameters: balancing cost and solve time

2026-11-03 · PushT, branch `cukernelsosTest`

Links:
- Online version of this doc (with the time-vs-cost chart): https://claude.ai/artifact/PMc2gxkT4y5jqjr3m7kAmW
- Interactive PushT cost surface viewer (plain cost and LSE σ = 0.001–0.3): https://claude.ai/artifact/5Hm1Xywk8v4Hpm8z9LS9RK

## Summary

With the current config (h 0.7, k 512, replace 0, plus a stop-condition fix), one PushT planning step takes **0.70 s** (CACS sampling 0.23 s + ADMM 0.44 s). The old config took 1.48 s. The solution cost is 8.89, compared with 8.53 before; two-level averages 14.30.

One rule explains almost every sweep: **more rounds give lower cost and take more time.** Each round is one GPU rollout call (≈22 ms fixed overhead), and each round re-centres the proposals on the cheapest samples found so far. batch, reuse and k all change the round count, so they all trade cost against time along the same line.

Current `CacsConfig` (`global_mppi/cacs.py`), used by both the closed loop and the single-step test:

| n | initial | reuse | h | max_clique | batch | k | global_every | replace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 200 | 40 | 0.7 | 20 | 768 | 512 | 125 | 0 |

## How a CACS call spends its time

Only the points that are kept get rolled out. Proposing, rule checks and ranking all run on the CPU and use costs that are already known.

1. **Initial batch.** Roll out `initial` = 200 scrambled Sobol points (1 GPU call). Up to `reuse` = 40 of them become graph seeds: first the cheapest 10%, then cheap points far from the seeds, each only if the graph rules allow it.
2. **Rounds**, until the graph holds n = 1000 samples:
    1. Propose `batch` = 768 points. 25% are global Sobol points; 75% are local steps of 0.24–1.03 h from a base sample (20% elite anchors, 35% low-degree tips, 45% cost-weighted). 20% of the directions are pulled toward an anchor.
    2. Rule check (geometry only): find the kernel neighbours S (KD-tree, distance < h), build the two-hop set T from G, and require |T| < `max_clique` and T to be a clique of H. This is set lookups, no costs.
    3. Rank the survivors with a proxy: cost of the nearest kept sample + 0.35 × distance (scaled by the cost spread).
    4. Pick up to `k` = 512, re-checking the rules after each pick, since every pick changes the graph.
    5. Roll out the picks in one GPU call (padded to k rows) and write the costs back. The next round's base weights, anchors and proxy use them.
3. **Replacement** (only when `replace` > 0; now off): drop samples from the costliest 30% and re-insert cheaper proposals.

The reverse insertion order is a perfect elimination order of H, so KSOS factors the samples in that order.

**Profile of the old config** (h 1.2, k 128, replace 40; seed 0; 569 ms in total):

| Part | Time | Share |
| --- | --- | --- |
| GPU rollouts: 14 round calls of 128 | 344 ms | 60% |
| GPU rollouts: initial batch | 28 ms | 5% |
| GPU rollouts: replacement batch | 24 ms | 4% |
| Rule checks on proposals (`batch_rule`) | 96 ms | 17% |
| Re-checks while picking | ~28 ms | 5% |
| Replacement CPU work | ~19 ms | 3% |
| Proposing + proxy ranking | ~13 ms | 2% |

Rollouts dominate because each GPU call carries about 22 ms of fixed overhead regardless of size. Measured rollout time per call: 1 point 21.7 ms, 128 points 25.5 ms, 512 points 29.8 ms, 1024 points 40.8 ms, 4096 points 76.9 ms. **So what speeds CACS up is fewer calls, not fewer points.** The new config needs about 5 calls instead of 16.

The profile also uncovered a bug in the pick loop: the stop test counted the round's picks twice, which created halving tail rounds (96, 48, 24, … picks). It is fixed (`global_mppi/cacs.py`, pick loop: `len(g.X) >= cfg.n`).

## Each hyperparameter: effect on cost and time

All numbers below come from CACS sampling alone (no KSOS) on PushT from the pusht.py start state. *best* is the cheapest sample (lower is better) and *top-20* is the mean of the 20 cheapest. The std of *best* across seeds is about 1.0–1.7, so with 10 seeds a difference under about 0.8 is noise.

| Parameter | Now | What it controls | Raise it | Lower it |
| --- | --- | --- | --- | --- |
| k | 512 | Points picked and rolled out per round (one GPU call) | Fewer rounds, faster up to ~512, then slower (padding) | More rounds, lower cost, slower |
| batch | 768 | Proposals per round (CPU only, no rollout) | No gain; more CPU rule checks above ~1024 | Fewer feasible points per round, so more rounds: lower cost, slower |
| reuse | 40 | Initial Sobol points kept as graph seeds (already rolled out) | Fewer rounds, faster down to ~0.17 s; cost no better | More rounds: lower cost, slower |
| h | 0.7 | Kernel radius; also sets the local step size | Coarser, costlier samples | Finer; below 0.6 cost rises again |
| n | 1000 | Samples kept for KSOS | Bigger KSOS problem, slower | Much worse cost |
| replace | 0 | Replacement moves after n samples | Small cost gain (in the noise), +~0.05 s | — |
| max_clique | 20 | Upper bound on H's clique size | No clear effect | Worse cost |
| initial | 200 | Initial Sobol rollouts | More rollouts, no gain | Fine down to 100 |
| global_every | 125 | One isolated global point per this many samples | No clear effect | No clear effect |

### k and h (10 seeds, with the stop fix, replace 0 unless noted)

| Config | Time | Rounds | best | top-20 |
| --- | --- | --- | --- | --- |
| h 0.7, k 128 | 0.293 s | 8.0 | **8.55** | 11.08 |
| h 0.7, k 192, replace 40 | 0.260 s | 5.4 | 9.19 | 12.10 |
| h 0.7, k 256, replace 40 | 0.251 s | 4.8 | 9.77 | 12.24 |
| **h 0.7, k 512 (now)** | **0.197 s** | 4.0 | 10.11 | 13.05 |
| h 1.2, k 512, replace 40 | 0.281 s | 5.0 | 9.64 | 13.37 |
| h 1.2, k 1024, replace 40 | 0.327 s | 5.0 | 9.64 | 13.37 |

Above k ≈ 512 the round count stops falling: picks conflict with each other inside a round, so at most a few hundred are accepted per round whatever k is. Each padded GPU call just gets more expensive.

h sweep (old code: k 128, replace 40, 10 seeds; time ≈ 0.50–0.54 s for all):

| h | 0.4 | 0.5 | 0.6 | **0.7** | 0.8 | 1.0 | 1.2 (old) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| best | 9.39 | 9.00 | 8.57 | **8.40** | 8.73 | 9.37 | 9.20 |
| top-20 | 10.24 | 10.13 | 10.11 | **10.11** | 10.75 | 11.75 | 12.70 |

h barely changes time but has the largest effect on cost: 0.6–0.7 is best. In the closed loop, h is scaled by `decay_rate**it` together with the trust-region box.

### batch (10 seeds, current config otherwise)

| batch | Time | Rounds | best | top-20 |
| --- | --- | --- | --- | --- |
| 512 | 0.248 s | 5.8 | **9.22** | **12.26** |
| 640 | 0.222 s | 4.9 | 9.99 | 12.93 |
| **768 (now)** | **0.195 s** | 4.0 | 10.11 | 13.05 |
| 1024 | 0.199 s | 3.9 | 10.08 | 13.51 |
| 1536 | 0.228 s | 3.9 | 10.42 | 13.26 |
| 2048 | 0.284 s | 4.6 | 9.94 | 13.45 |

Only the 75% of proposals that are local steps compete for picks. The 25% global Sobol proposals are used only when an isolated global point is due (about 7 per call), so most of them are generated and discarded.

### reuse (10 seeds, current config otherwise)

| reuse | Time | Rollouts | Rounds | best | top-20 |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.323 s | 1190 | 7.7 | 9.29 | 11.64 |
| 20 | 0.246 s | 1180 | 5.4 | **8.84** | **11.08** |
| **40 (now)** | 0.196 s | 1160 | 4.0 | 10.11 | 13.05 |
| 80 | 0.183 s | 1120 | 3.6 | 9.95 | 14.00 |
| 120 | **0.169 s** | 1080 | 3.1 | 9.81 | 13.38 |
| 200 | 0.172 s | 1000 | 3.0 | 9.92 | 13.73 |
| 200, initial 400 | 0.195 s | 1200 | 2.9 | 9.50 | 12.81 |

Extra seeds are uniform Sobol points, and most of them are not cheap. They fill the graph without new rollouts (faster), but they take slots that cost-guided rounds would have filled.

### n, replace, max_clique, initial, global_every (old code: h 1.2, k 128; 5 seeds)

| Change from old default | Time | Rollouts | best | top-20 |
| --- | --- | --- | --- | --- |
| old default | 0.539 s | 1200 | 9.03 | 12.39 |
| n 500 | 0.372 s | 700 | 10.88 | 14.96 |
| n 250 | 0.290 s | 450 | 11.70 | 16.30 |
| replace 0 | 0.498 s | 1160 | 9.03 | 12.46 |
| replace 160 | 0.602 s | 1320 | 8.73 | 12.00 |
| max_clique 10 | 0.557 s | 1200 | 9.75 | 13.99 |
| max_clique 30 | 0.562 s | 1200 | 9.48 | 12.38 |
| initial 100 | 0.538 s | 1100 | 9.03 | 11.88 |
| initial 400 | 0.549 s | 1400 | 9.16 | 12.02 |
| global_every 60 | 0.549 s | 1200 | 9.03 | 12.26 |
| global_every 250 | 0.536 s | 1200 | 8.85 | 12.16 |

With the fix and k 512, n 500 (h 0.6, replace 0) takes 0.105 s, but best rises to 11.23.

## Single-step results with KSOS

The new config halves CACS + KSOS time (1.48 s → 0.70 s) and keeps the solution far below two-level (8.89 vs 14.30). Its solution cost is 0.36 higher than with the old config.

The setup is one planning step from the pusht.py start state, run through the closed-loop code path (`GlobalMPPI.solve_ksos` → `_solve_cacs`), with 5 controller seeds. *solution* is the cost of the plan after the guard, from a fresh rollout. Two-level uses the same number of rollouts as CACS, clusters of 10, the same ADMM options and the same guard. Sobol is the best of the same number of plain Sobol points, without KSOS.

| Mean of 5 seeds | Old config (h 1.2, k 128, replace 40) | New config (h 0.7, k 512, replace 0) | Two-level | Sobol best |
| --- | --- | --- | --- | --- |
| Rollouts | 1200 | 1160 | 1160 | 1160 |
| Sampling | 0.56 s | **0.23 s** | 0.05 s | — |
| ADMM | 0.89 s | **0.44 s** | ~0.09 s | — |
| Total | 1.48 s | **0.70 s** | ~0.16 s | — |
| ADMM status | 4 converged, 1 time limit | 5 converged | 5 converged | — |
| Solution | **8.53** | 8.89 | 14.30 | 11.09 |

Per seed, new config:

| Seed | Max clique (ADMM) | best sample | Solution | ADMM iters | Sampling | ADMM | Two-level solution | Sobol best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | 9.08 | 9.08 | 170 | 0.206 s | 0.439 s* | 14.45 | 14.42 |
| 1 | 20 | 9.26 | 9.09 | 210 | 0.208 s | 0.430 s | 10.02 | 11.32 |
| 2 | 20 | 9.55 | 9.55 | 180 | 0.320 s | 0.433 s | 13.66 | 10.80 |
| 3 | 23 | 8.20 | 8.20 | 90 | 0.209 s | 0.353 s | 17.84 | 8.74 |
| 4 | 21 | 8.61 | 8.51 | 290 | 0.205 s | 0.535 s | 15.54 | 10.16 |

\*In the saved run (`cacs_single_step_new.json`), seed 0's ADMM took 1.94 s. A rerun took 0.44 s with the same result, so the 1.94 s was a one-off; the mean uses the rerun.

Notes:

- KSOS adds little over the best sample: the solution equals the best sample or is slightly below it (seeds 1 and 4). The quality of the CACS samples is what sets the result.
- ADMM reports max cliques of 20–23 even though CACS keeps H's cliques below 20, because ADMM builds its own chordal extension instead of reusing H.
- The solution is 0.36 higher than with the old config, which is within seed noise. The sweeps suggest the gap is real but small (fewer rounds = less cost guidance).
- The closed loop builds `CacsConfig(n=ksos_num_samples)` (`global_mppi/algs/global_mppi.py:187`), so it already runs the new config. The closed-loop logs under `closed_loop/seeds/globalmppi_cacs*` predate it and should be rerun.

## Recommended presets and open issues

Pick a preset by how much time a planning step can spend. Every move from *Current* toward lower cost adds rounds and about 0.05–0.1 s. No setting is both fastest and cheapest.

| Preset | Change from current | Sampling time | best | top-20 | When to use |
| --- | --- | --- | --- | --- | --- |
| Fastest | reuse 120 | 0.169 s | 9.81 | 13.38 | Tight control rate |
| Current | — | 0.196 s | 10.11 | 13.05 | Default speed/cost balance |
| Balanced | reuse 20 | 0.246 s | 8.84 | 11.08 | Lower cost for +0.05 s |
| Lowest cost | k 128 | 0.293 s | 8.55 | 11.08 | Cost matters more than time |

The 0.3 difference in *best* between Fastest and Current is noise, so Fastest gives the same quality, just faster. Combinations of these changes (for example reuse 20 + batch 512) are untested.

Open issues:

- [ ] Confirm *Balanced* in the single-step test with KSOS: a lower best sample does not guarantee a lower solution.
- [ ] Rerun the closed-loop CACS seeds with the current config; the saved logs use the old one.
- [ ] CACS is seeded with the call counter (`seed=self._cacs_calls`), so the initial Sobol batch is identical across controller seeds. Seed it from the controller seed instead.
- [ ] Most of the 25% global proposals per round are discarded (an isolated global point is used only once every 125 samples). Either let global points with neighbours compete for picks, or shrink the share.
- [ ] ADMM builds its own chordal extension (cliques 20–23) instead of using H from CACS (cliques < 20).

## Test setup and reproduction

All tests use the PushT start state from pusht.py (block 0.1, 0.1, 1.3; pusher 0, 0), a 1 s horizon, 6 cubic knots × 2 controls (12-D box [−1, 1]^12), and branch `cukernelsosTest`. JIT compile time is excluded: each script warms up every rollout shape first. In the sweeps, seeds differ per run: `cacs_sample(seed=s, rng=default_rng(s + 101))`.

| Script (`examples/sample_method_test/`) | What it measures | Results |
| --- | --- | --- |
| `cacs_sweep.py --stage oat` | CACS sampling only, one parameter at a time | `cacs_sweep_oat.json` |
| `cacs_sweep.py --stage configs --configs '[...]'` | CACS sampling only, full configs | `cacs_sweep_h.json`, `_combo`, `_speed`, `_fixed`, `_n1000`, `_new_default`, `_batch`, `_reuse` |
| `pusht_cacs_budgeted_gpu.py --seeds 0 1 2 3 4` | One planning step: CACS + KSOS vs two-level vs Sobol, closed-loop code path | `cacs_single_step.json` (old config), `cacs_single_step_new.json` (new config) |

Example:

```bash
cd examples/sample_method_test
python cacs_sweep.py --stage configs --configs '[{"reuse":20},{}]' --seeds 0 1 2 3 4 5 6 7 8 9 --json out.json
python pusht_cacs_budgeted_gpu.py --seeds 0 1 2 3 4 --json cacs_single_step_new.json
```

Sweeps run before the stop fix (`_oat`, `_h`, `_combo`, `_speed`) overstate the time by roughly 0.15–0.2 s because of the halving tail rounds; their cost comparisons still hold.

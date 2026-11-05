# CACS closed-loop hyperparameter sweep (PushT)

2026-11-04 · PushT closed loop, branch `cukernelsosTest`

## Summary

CACS hyperparameters barely trade time for cost in closed loop. Every config runs at 6.85–7.96 s per cycle. Final cost differs between configs by less than it differs between seeds of the same config (more than 10×). Keep the main config for the paper.

| Use | Config | Result |
| --- | --- | --- |
| Main config | reuse 40, k 512, h 0.7, batch 768, initial 200 | 6 seeds: 2.1e-4 ± 1.8e-4, 7.31 s per cycle |
| Fast config | reuse 120, rest as main | 6 seeds: 2.9e-4 ± 1.8e-4, 6.98 s per cycle |
| Candidate, not yet confirmed | batch 512, rest as main | seed 0 only: 7.0e-5, 7.66 s per cycle |

All configs use n = 1000 samples and no replacement.

## Setup

- Task: PushT. Start state: block (0.1, 0.1, 1.3), pusher (0, 0). 100 control cycles at 50 Hz.
- GlobalMPPI: CACS ×5 restarts with warm start, in shrinking boxes (decay 0.8 per restart; h shrinks with the box). Then KSOS on ADMM-cuda (time limit 2 s, λ 1e-5), then MPPI 1000 × 5 with σ 0.1.
- Fixed CACS settings: n 1000, max_clique 20, global_every 125, replace 0.
- Swept settings, one at a time from the main config: h, k, initial, batch, reuse.
- Command (overrides via `--cacs` JSON):

```bash
python examples/pusht.py --no-wandb --headless --seed 0 --ksos-solver ADMM-cuda --ksos-sampling cacs \
  --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 --cacs '{"batch": 512}' globalmppi
```

- Final cost = cost at cycle 100. Plan time = mean over cycles 1–100 (cycle 0 is JIT). std uses ddof = 1.

## 6-seed results (seeds 0–5)

| Method | Final cost, mean ± std | Geometric mean | Median | Mean cost over run | Plan time / cycle | CACS / restart |
| --- | --- | --- | --- | --- | --- | --- |
| **CACS ×5 main (reuse 40, k 512)** | **2.1e-4 ± 1.8e-4** | **1.3e-4** | **1.7e-4** | **0.89** | 7.31 ± 0.32 s | 0.24 s |
| CACS ×5 reuse 120 | 2.9e-4 ± 1.8e-4 | 2.1e-4 | — | 0.92 | **6.98 ± 0.38 s** | **0.17 s** |
| CACS ×5 old cfg (h 1.2, k 128, replace 40) | 2.2e-4 ± 1.7e-4 | 1.4e-4 | — | 0.97 | 8.42 ± 0.42 s | 0.42 s |
| CACS ×5 k 128 | 3.6e-4 ± 2.8e-4 | 2.2e-4 | — | 1.04 | 7.86 ± 0.20 s | 0.36 s |
| DIAL (2000 × 5) | 2.2e-2 ± 1.3e-2 | — | 2.2e-2 | — | 0.36 s | — |
| MPPI (2000 × 5) | 3.4e-1 ± 8.3e-2 | — | 3.8e-1 | — | 0.36 s | — |
| PS (2000 × 5) | 5.9e-1 ± 1.4 | — | 1.5e-3 | — | 0.36 s | — |
| GCS (one run) | 1e-5 | — | — | — | 120.44 s | — |

Final cost per seed:

| Seed | main (k 512) | reuse 120 | old cfg | k 128 |
| --- | --- | --- | --- | --- |
| 0 | 4.1e-4 | 4.0e-4 | 3.9e-4 | 2.7e-5 |
| 1 | 2.4e-4 | 9.8e-5 | 2.1e-4 | 7.0e-5 |
| 2 | 3.1e-5 | 4.2e-4 | 4.2e-4 | 4.9e-4 |
| 3 | 9.6e-5 | 3.8e-4 | 2.5e-5 | 3.1e-4 |
| 4 | 3.6e-5 | 4.1e-4 | 2.4e-4 | 4.8e-4 |
| 5 | 4.2e-4 | 3.1e-5 | 3.9e-5 | 7.8e-4 |

- k 128 vs main: k 128 has higher cost (not significant, Mann–Whitney p = 0.39) and is 0.55 s slower per cycle. Dropped.
- The earlier "CACS ×5 old (to 81)" result (3.8e-5) was one seed on code with a pick-loop bug (about 14 rounds per restart instead of 8), and it stopped at cycle 80. Rerun over 6 seeds on the fixed code, the old config is no better than main.

## Seed-0 sweep (one change from the main config)

| Group | Change | Final cost | Mean cost over run | Plan time / cycle | CACS / restart |
| --- | --- | --- | --- | --- | --- |
| — | **main** | 4.1e-4 | 0.886 | 7.76 s | 0.243 s |
| h | h 0.5 | 8.0e-4 | 0.958 | 6.95 s | 0.226 s |
| h | h 0.6 | 5.0e-4 | 0.821 | 7.48 s | 0.223 s |
| h | h 0.8 | 6.8e-4 | 0.813 | **6.85 s** | 0.242 s |
| k | k 128 | **2.7e-5** | 1.456 | 7.96 s | 0.365 s |
| k | k 256 | 5.8e-4 | 1.063 | 7.03 s | 0.260 s |
| rollouts | initial 512 | 3.7e-4 | 0.882 | 7.27 s | 0.202 s |
| rollouts | initial 1024 | **4.6e-5** | 0.994 | 7.16 s | 0.247 s |
| rollouts | initial 2048 | 4.8e-4 | 0.872 | 7.26 s | 0.255 s |
| rounds | batch 512 | **7.0e-5** | 0.860 | 7.66 s | 0.253 s |
| rounds | reuse 20 | 6.5e-4 | 1.060 | 7.53 s | 0.275 s |
| rounds | reuse 20 + batch 512 | 6.3e-4 | **0.802** | 7.45 s | 0.294 s |
| rounds | reuse 120 | 4.0e-4 | 1.098 | 7.09 s | **0.168 s** |

## Analysis

- **Time barely moves.** ADMM takes about 1.1 s per restart (5 restarts per cycle), while CACS takes 0.17–0.37 s. CACS settings can only save or add a few tenths of a second per cycle.
- **Seed noise dominates cost.** The same config varies by more than 10× across seeds. A 2–3× difference on one seed means nothing; only about 10× is worth a 6-seed check. k 128 shows why: best on seed 0, worse than main over 6 seeds.
- **h:** 0.7 is the best of 0.5–0.8. Keep it.
- **k:** no monotone trend; k 512 stays.
- **initial (more rollouts):** only 1024 stands out; 512 and 2048 are no better than main, so 1024 is likely a lucky seed.
- **batch:** 512 gave a low final cost and a slightly better mean cost than main at about the same time. It is the only setting that may lower cost without adding time.
- **reuse:** the time knob. reuse 120 is the fastest config (CACS 0.17 s per restart) at similar cost; reuse 20 is worse.
- **Which parameter balances time and cost:** reuse sets the time (40 for cost, 120 for speed). h 0.7 and k 512 are already the best settings. batch 512 is the one candidate for lower cost at no extra time.

## Next steps

- Run 6 seeds of batch 512 and initial 1024 (about 1.5 h each). If batch 512 averages clearly below 2.1e-4, make it the main config.
- Startup crashes (segfault or abort during JAX compile at cycle 0) are frequent, up to 3 in a row; each retry then finishes all 100 cycles. They do not affect results.

## Files

- Logs: `examples/single_run_test/PushT/closed_loop/seeds/globalmppi_cacs5<tag>_seed<k>.log`. Tags: `new` (main), `r120`, `old`, `k128`, `k256`, `r20`, `b512`, `r20b512`, `i512`, `i1024`, `i2048`, `h05`, `h06`, `h08`.
- Figures in `examples/single_run_test/PushT/closed_loop/`:
  - `pusht_closed_loop_6seeds_main.png`: PS, MPPI, DIAL, CACS ×5 main, GCS (`plot_closed_loop_seeds.py six_seeds_main … --gcs`).
  - `pusht_closed_loop_6seeds.png`: all 6-seed CACS configs.
  - `pusht_closed_loop_paper.png`: paper style, median + 25–75% band (`plot_paper_style.py`).
- Single-step hyperparameter study: `docs/cacs_hyperparameters.md`.

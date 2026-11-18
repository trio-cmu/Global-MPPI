# Figure: Sobol vs CACS sampling and the aggregate sparsity of the KSOS SDP

File: `sobol_vs_cacs_2d_paper.png`, made by `plot_cacs_2d.py --max-clique 15` (100 samples, seed 0, h = 0.17).
Layout: column (a) Sobol sampling, column (b) CACS sampling; top row the samples, bottom row the aggregate sparsity pattern.

## Caption (draft)

**Sobol vs. CACS sampling on a 2-D five-basin cost** (100 samples each, kernel radius h = 0.17). Top: samples
(darker = lower cost; star = global minimum). Bottom: aggregate sparsity pattern of M(α) = λI + Φ diag(α) Φᵀ in
RCM order; red marks the largest clique, i.e. the largest PSD block in ADMM. **(a)** Sobol sampling provides
uniform coverage, resulting in a maximum clique size of 21 that cannot be controlled: it is set by the sample
spacing and grows with the number of samples. **(b)** CACS (max_clique = 15) concentrates
samples in low-cost basins; shaded regions are cliques of its chordal certificate, and the largest clique stays
at the bound (15).

## Paragraph for the text (draft)

Figure X illustrates the difference on a 2-D function with five basins. Plain Sobol sampling (Fig. X(a)) spreads
its budget uniformly, so most samples land in high-cost regions, and the coupling between samples is dictated only
by their spacing: with the same kernel radius, its aggregate pattern is a wide band whose largest clique, 21,
grows without bound as samples are added (with h = 0.09: 4 at 100 samples, 93 at 800, 209 at 1,600). CACS
(Fig. X(b)) instead grows its samples from cheap ones and admits a candidate only if it keeps the certificate H
chordal with cliques of at most max_clique vertices. Its samples concentrate in the low-cost basins, the best
sample is closer to the global minimum (0.012 vs 0.021), and its pattern decomposes into small blocks whose
largest clique equals the prescribed bound of 15. ADMM therefore handles PSD blocks of bounded size, regardless of
how many samples are used.

## Numbers behind the figure

| | (a) Sobol | (b) CACS (max_clique 15) |
|---|---|---|
| samples | 100 | 100 (115 cost evaluations) |
| best sample (global minimum 0.010) | 0.021 | 0.012 |
| pairs in E | 1,252 | 537 |
| cliques | 64 | 34 |
| largest clique | 21 | 15 |
| chordal fill | 0 | 0 |
| largest clique at 1,600 samples (h = 0.09 test) | 209 | stays at max_clique |

## Points to be accurate about

- The CACS pattern in (b) builds Φ with the CACS samples in RCM order, the same ordering as (a), so the two
  matrices are comparable. The clique bound is *guaranteed* in the order the closed-loop solver uses (reverse
  insertion order, `cacs_2d_sparsity.png`: 546 pairs, largest clique 15, no fill). In RCM order it holds for this
  seed (15) but is not guaranteed in general (it exceeded the bound on other seeds). If you want (b) to show
  exactly the decomposition the solver uses, with a banded look, relabel that E with the RCM order of H instead of
  rebuilding Φ (`plot_cacs_h_rcm_2d.py`): the cliques then stay identical.
- CACS uses 115 cost evaluations to keep 100 samples (20 initial Sobol points plus picks); Sobol uses 100.
- The best-sample comparison depends on the seed (CACS was better on 5 of 7 seeds at this setting); report it as
  an illustration, not a statistic.
- 13 CACS samples lie on the box edge because proposals are clipped to the box.

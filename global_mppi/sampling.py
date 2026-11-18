"""Sample designs for the KSOS global search over flattened control knots.

A sample is one knot sequence flattened to a vector of dim = num_knots * nu
(reshape to (num_knots, nu) for a rollout), drawn in a box [lower, upper]^dim:
the control range, or a trust region inside it. The samplers do not know the
horizon, so the same code serves short and long horizons.

- SobolSampler: n scrambled Sobol points over the box (GPU, jittable).
- TwoLevelSampler: two-level Sobol design (GPU, jittable). n / cluster_size
  Sobol cluster centers, cluster_size Sobol points uniform in a ball of
  cluster_radius * sigma around each, sigma = sigma_factor x the centers'
  mean nearest-neighbour distance. With the defaults (0.5, 0.5) points in one
  cluster are within sigma of each other, so every cluster is a clique of the
  Wendland kernel matrix K and, when clusters stay more than sigma apart, the
  largest clique of K is cluster_size.
- two_level_cpu: ksos-tools' version of the same design (fixed seeds, CPU),
  on a box centered at 0.

Only the Sobol unit points are made on the CPU (scipy has the Sobol
generator, JAX does not), once, in the constructor; `sample` then shifts,
scales and clips them on the GPU and can be called inside jax.jit.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import norm, qmc


class SobolSampler:
    """num_samples scrambled Sobol points in [lower, upper]^dim.

    Each call applies a random shift (mod 1) to the fixed Sobol unit points
    (a Cranley-Patterson rotation: still a low-discrepancy set, but a new one
    per rng), then scales them into the box.
    """

    def __init__(self, num_samples: int, dim: int, seed: int = 0) -> None:
        self.num_samples = num_samples
        self.dim = dim
        self.unit = jnp.asarray(
            qmc.Sobol(d=dim, scramble=True, seed=seed).random(num_samples),
            dtype=jnp.float32,
        )

    def sample(self, rng: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
        """Samples, (num_samples, dim). lower / upper: scalars or (dim,)."""
        u = jnp.mod(self.unit + jax.random.uniform(rng, (self.dim,)), 1.0)
        return lower + (upper - lower) * u


def sobol_ball(count: int, dim: int, seed: int) -> np.ndarray:
    """count scrambled Sobol points uniform in the unit dim-ball (ksos-tools'
    sobol_ball): a (dim + 1)-dimensional Sobol point gives the direction
    (normal quantiles of the first dim coordinates) and the radius u^(1/dim)
    (the last one)."""
    u = qmc.Sobol(dim + 1, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]
    u = np.clip(u, 1e-12, 1 - 1e-12)
    z = norm.ppf(u[:, :dim])
    return z / np.linalg.norm(z, axis=1, keepdims=True) * u[:, dim:] ** (1.0 / dim)


class TwoLevelSampler:
    """Two-level Sobol design in [lower, upper]^dim; see the module docstring.

    Both levels are Sobol: the centers are a scrambled Sobol set in the box,
    and each cluster has its own scrambled Sobol set in the unit ball (made
    once, in the constructor, like the centers; as in ksos-tools'
    two-level design). Each call shifts the centers randomly (mod 1); the
    ball points around each center are fixed, scaled to the ball.

    `cluster` gives each sample's cluster (samples come cluster by cluster);
    the last cluster is partial if cluster_size does not divide num_samples.
    """

    def __init__(
        self,
        num_samples: int,
        dim: int,
        cluster_size: int = 10,
        seed: int = 0,
        sigma_factor: float = 0.5,
        cluster_radius: float = 0.5,
    ) -> None:
        self.num_samples = num_samples
        self.dim = dim
        self.cluster_size = cluster_size
        self.sigma_factor = sigma_factor
        self.cluster_radius = cluster_radius
        self.num_clusters = -(-num_samples // cluster_size)
        self.centers_unit = jnp.asarray(
            qmc.Sobol(d=dim, scramble=True, seed=seed).random(self.num_clusters),
            dtype=jnp.float32,
        )
        # second level: one Sobol set in the unit ball per cluster, (m, c, dim)
        self.ball_unit = jnp.asarray(
            np.stack([sobol_ball(cluster_size, dim, seed=1000 * (seed + 1) + k)
                      for k in range(self.num_clusters)]),
            dtype=jnp.float32,
        )
        self.cluster = np.arange(num_samples) // cluster_size
        self._no_self = jnp.diag(jnp.full(self.num_clusters, jnp.inf, dtype=jnp.float32))

    def sample(
        self, rng: jax.Array, lower: jax.Array, upper: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Samples (num_samples, dim), the kernel radius sigma, and the
        fraction of coordinates clipped into the box. lower / upper: scalars
        or (dim,)."""
        # first level: Sobol centers, randomly shifted (mod 1), in the box
        u = jnp.mod(self.centers_unit + jax.random.uniform(rng, (self.dim,)), 1.0)
        centers = lower + (upper - lower) * u
        # sigma = sigma_factor x the centers' mean nearest-neighbour distance
        d2 = jnp.sum((centers[:, None] - centers[None]) ** 2, axis=-1) + self._no_self
        sigma = self.sigma_factor * jnp.mean(jnp.sqrt(jnp.min(d2, axis=1)))
        # second level: each cluster's Sobol ball points, scaled to the ball
        # of cluster_radius * sigma
        offsets = self.cluster_radius * sigma * self.ball_unit
        points = (centers[:, None] + offsets).reshape(-1, self.dim)[: self.num_samples]
        # a ball can reach past the box; clipping never stretches a distance,
        # so every cluster stays a clique
        clipped = jnp.mean((points < lower) | (points > upper))
        return jnp.clip(points, lower, upper), sigma, clipped


def two_level_cpu(
    num_samples: int,
    dim: int,
    cluster_size: int = 10,
    radius: float = 1.0,
    sigma: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """ksos-tools' two-level design (fixed seeds, CPU) on [-radius, radius]^dim:
    (samples clipped into the box, sigma, cluster, fraction of coordinates
    clipped). sigma None uses ksos-tools' rule (0.5 x the centers' mean
    nearest-neighbour distance)."""
    from ksos_tools.problems.sampling import Sampling, make_samples

    design = Sampling(kind="clustered", order="none", sigma=sigma, cluster_size=cluster_size)
    samples, sigma = make_samples(num_samples, design, dim=dim, radius=radius)
    clipped = float(np.mean(np.abs(samples) > radius))
    return np.clip(samples, -radius, radius), float(sigma), np.arange(num_samples) // cluster_size, clipped


@dataclass
class ClusterReport:
    """How a two-level sample set (and its costs) came out; see cluster_report."""

    num_clusters: int
    cluster_size: int
    sigma: float
    within: float  # largest distance inside one cluster
    between: float  # smallest distance between two clusters
    links_between: int  # kernel links (pairs within sigma) across clusters
    plateau: float  # the median cost (PushT: the "block never touched" cost)
    off_plateau: int  # samples whose cost differs from the plateau
    clusters_off_plateau: int  # clusters with at least one such sample
    per_cluster: np.ndarray  # (num_clusters, 3): cost min, median, max

    def __str__(self) -> str:
        lines = [
            f"two-level: {self.num_clusters} clusters of {self.cluster_size}; "
            f"within-cluster distance <= {self.within:.3f}, "
            f"between-cluster distance >= {self.between:.3f} (sigma {self.sigma:.3f}); "
            f"{self.links_between} links between clusters",
            f"  clusters with a sample off the plateau ({self.plateau:.3f}): "
            f"{self.clusters_off_plateau} of {self.num_clusters}; "
            f"samples off the plateau: {self.off_plateau}",
        ]
        if self.num_clusters <= 20:
            lines += [
                f"  cluster {c}: cost min {lo:7.3f}  median {med:7.3f}  max {hi:7.3f}"
                for c, (lo, med, hi) in enumerate(self.per_cluster)
            ]
        return "\n".join(lines)


def cluster_report(samples, costs, cluster, sigma) -> ClusterReport:
    """Check a two-level sample set against its design: every cluster within
    sigma (a clique of K), no kernel links between clusters; and summarize the
    costs per cluster. Uses a k-d tree, never an n x n distance matrix."""
    samples, costs, cluster = np.asarray(samples), np.asarray(costs), np.asarray(cluster)
    num_clusters = int(cluster.max()) + 1
    cluster_size = int(np.bincount(cluster).max())
    kd = cKDTree(samples)
    pairs = kd.query_pairs(sigma, output_type="ndarray")
    links_between = int(np.sum(cluster[pairs[:, 0]] != cluster[pairs[:, 1]]))
    within = max(
        (pdist(samples[cluster == c]).max() for c in range(num_clusters) if np.sum(cluster == c) > 1),
        default=0.0,
    )
    # the nearest point of another cluster is among the cluster_size + 1 nearest
    dist, idx = kd.query(samples, k=min(cluster_size + 1, len(samples)))
    between = float(np.min(np.where(cluster[idx] != cluster[:, None], dist, np.inf)))
    plateau = float(np.median(costs))
    off = ~np.isclose(costs, plateau, atol=1e-3)
    per_cluster = np.array(
        [[np.min(fc), np.median(fc), np.max(fc)] for fc in (costs[cluster == c] for c in range(num_clusters))]
    )
    return ClusterReport(
        num_clusters=num_clusters,
        cluster_size=cluster_size,
        sigma=float(sigma),
        within=float(within),
        between=between,
        links_between=links_between,
        plateau=plateau,
        off_plateau=int(off.sum()),
        clusters_off_plateau=len(np.unique(cluster[off])),
        per_cluster=per_cluster,
    )

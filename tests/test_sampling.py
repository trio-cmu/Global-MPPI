import jax
import jax.numpy as jnp
import numpy as np
import pytest

from global_mppi.sampling import (
    SobolSampler,
    TwoLevelSampler,
    cluster_report,
    two_level_cpu,
)


def test_sobol_sampler() -> None:
    """Samples fill the box, differ per rng, and work inside jit."""
    sampler = SobolSampler(64, 12, seed=0)
    lower, upper = -jnp.ones(12), 2.0 * jnp.ones(12)
    sample = jax.jit(lambda key: sampler.sample(key, lower, upper))
    a = sample(jax.random.PRNGKey(0))
    b = sample(jax.random.PRNGKey(1))
    assert a.shape == (64, 12)
    assert jnp.all(a >= lower) and jnp.all(a <= upper)
    assert not jnp.allclose(a, b)
    assert jnp.allclose(a, sample(jax.random.PRNGKey(0)))


@pytest.mark.parametrize("dim", [12, 36])  # short and long PushT horizons
def test_two_level_sampler(dim: int) -> None:
    """Each cluster is a clique of K (within sigma) and inside the box."""
    sampler = TwoLevelSampler(200, dim, cluster_size=10, seed=0)
    lower, upper = -jnp.ones(dim), jnp.ones(dim)
    points, sigma, clipped = jax.jit(lambda key: sampler.sample(key, lower, upper))(
        jax.random.PRNGKey(0)
    )
    assert points.shape == (200, dim)
    assert jnp.all(points >= lower) and jnp.all(points <= upper)
    assert 0.0 <= float(clipped) < 1.0
    np.testing.assert_array_equal(sampler.cluster, np.repeat(np.arange(20), 10))
    report = cluster_report(np.asarray(points), np.zeros(200), sampler.cluster, float(sigma))
    assert report.num_clusters == 20 and report.cluster_size == 10
    assert report.within <= float(sigma)


def test_two_level_partial_cluster() -> None:
    """A cluster size that does not divide n leaves the last cluster short."""
    sampler = TwoLevelSampler(25, 4, cluster_size=10)
    points, _, _ = sampler.sample(jax.random.PRNGKey(0), -1.0, 1.0)
    assert points.shape == (25, 4)
    assert np.bincount(sampler.cluster).tolist() == [10, 10, 5]


def test_two_level_cpu() -> None:
    """ksos-tools' design: clipped into the box, clusters within sigma."""
    samples, sigma, cluster, clipped = two_level_cpu(100, 12, cluster_size=10)
    assert samples.shape == (100, 12)
    assert np.all(np.abs(samples) <= 1.0)
    assert 0.0 <= clipped < 1.0
    report = cluster_report(samples, np.arange(100.0), cluster, sigma)
    assert report.within <= sigma
    assert report.per_cluster.shape == (10, 3)


def test_two_level_second_level_is_sobol() -> None:
    """The in-ball points are each cluster's Sobol ball set, scaled to the
    ball of cluster_radius * sigma around the cluster's center."""
    sampler = TwoLevelSampler(40, 12, cluster_size=10, seed=0, cluster_radius=0.01)
    points, sigma, clipped = sampler.sample(jax.random.PRNGKey(3), -1.0, 1.0)
    assert float(clipped) == 0.0  # small balls: nothing clipped
    X = np.asarray(points).reshape(4, 10, 12)
    unit = np.asarray(sampler.ball_unit)
    radius = sampler.cluster_radius * float(sigma)
    offsets = X - (X - unit * radius).mean(axis=1, keepdims=True)  # center from any point
    np.testing.assert_allclose(offsets, unit * radius, atol=1e-6)
    assert np.all(np.linalg.norm(unit, axis=-1) <= 1.0 + 1e-6)

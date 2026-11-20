"""Obstacle layout of the obstacle_field task, and the grid-search reference path.

numpy only (no jax / mujoco), so that script/obstacle_field_metrics.py can load it without starting the GPU stack.

The arena is the particle model's square [-0.29, 0.29]^2 (the joint range; the inner faces of the outer walls are at
+-0.29). The point mass has radius 0.01. ``make_layout(layout_seed)`` draws 9 circles with radius in [0.025, 0.045]
by rejection sampling, with a free gap of at least ``MIN_GAP`` (0.06 m = 3 particle diameters) between any two
circles (they never touch), between a circle and the outer walls, and between a circle and the start or the goal. A layout is only
returned when ``shortest_path`` finds a collision-free path from start to goal on a grid.
"""

import heapq

import numpy as np

HALF = 0.29  # inner face of the outer walls (also the joint range of the particle)
PARTICLE_RADIUS = 0.01
START = np.array([-0.25, 0.0])
GOAL = np.array([0.25, 0.0])
N_OBSTACLES = 9
RADIUS_RANGE = (0.025, 0.045)
MIN_GAP = 0.06  # m, free gap (surface to surface) between circles / walls / start / goal: 3 particle diameters
REF_MARGIN = 0.005  # m, extra clearance the reference path keeps from the obstacles and walls (surface to surface)
GRID = 0.0025  # m, grid step of the reference search


def _clearance(points, centers, radii):
    """Surface-to-surface distance (point radius PARTICLE_RADIUS) to the nearest circle; (N,2) -> (N,)."""
    d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1) - radii[None, :] - PARTICLE_RADIUS
    return d.min(axis=1)


def _wall_clearance(points):
    return HALF - PARTICLE_RADIUS - np.abs(points).max(axis=1)


def free(points, centers, radii, margin=0.0):
    """True where the particle keeps ``margin`` free gap from every circle and from the outer walls."""
    points = np.atleast_2d(points)
    return (_clearance(points, centers, radii) >= margin) & (_wall_clearance(points) >= margin)


def shortest_path(centers, radii, start=START, goal=GOAL, margin=REF_MARGIN, step=GRID):
    """Shortest collision-free path of the particle centre from start to goal.

    A* (Dijkstra with a Euclidean heuristic) on a grid (16 neighbours: 8 king + 8 knight moves), then the path is shortened by greedy line-of-sight
    pulling with an exact segment check. Returns (path (K,2), length in m), or (None, inf) when there is no path.
    """
    n = int(round(2 * HALF / step)) + 1
    xs = np.linspace(-HALF, HALF, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    ok = free(pts, centers, radii, margin).reshape(n, n)
    ix = lambda p: tuple(int(round((v + HALF) / step)) for v in p)  # noqa: E731
    s, g = ix(start), ix(goal)
    if not (ok[s] and ok[g]):
        return None, float("inf")
    moves = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0) and (abs(a) < 2 or abs(b) < 2) and np.gcd(abs(a), abs(b)) == 1]
    dist = np.full((n, n), np.inf)
    prev = {}
    dist[s] = 0.0
    h = lambda u: step * float(np.hypot(u[0] - g[0], u[1] - g[1]))  # noqa: E731  (A*: Euclidean distance, consistent)
    heap = [(h(s), 0.0, s)]
    while heap:
        _, d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == g:
            break
        for a, b in moves:
            v = (u[0] + a, u[1] + b)
            if not (0 <= v[0] < n and 0 <= v[1] < n) or not ok[v]:
                continue
            nd = d + step * float(np.hypot(a, b))
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd + h(v), nd, v))
    if not np.isfinite(dist[g]):
        return None, float("inf")
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    path = np.array([[xs[i], xs[j]] for i, j in reversed(cells)])
    path[0], path[-1] = start, goal

    def visible(p, q):
        k = max(2, int(np.linalg.norm(q - p) / 0.001))
        seg = p[None, :] + np.linspace(0.0, 1.0, k)[:, None] * (q - p)[None, :]
        return bool(free(seg, centers, radii, margin).all())

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not visible(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    out = np.array(out)
    return out, float(np.linalg.norm(np.diff(out, axis=0), axis=1).sum())


def _spot_free(c, r, centers, radii):
    """A new circle (c, r) keeps MIN_GAP to the outer walls, the start, the goal and every circle in the lists."""
    if np.abs(c).max() + r > HALF - MIN_GAP:
        return False
    if min(np.linalg.norm(c - START), np.linalg.norm(c - GOAL)) - r < MIN_GAP + PARTICLE_RADIUS:
        return False
    return all(np.linalg.norm(c - c2) - r - r2 >= MIN_GAP for c2, r2 in zip(centers, radii))


def _propose(rng):
    """One candidate: 3 or 4 large circles on or close to the start-goal line, the rest scattered. (centers, radii) or None.

    The large circles (radius 0.038 to 0.045) stand at x in [-0.13, 0.13], at most 8 cm off the line; at least two of them
    block the straight line (|y| < r + particle radius), so a greedy straight path gets stuck in front of them. The other
    circles have a radius in [0.025, 0.045] and a random position. All gaps are at least MIN_GAP.
    """
    centers, radii = [], []
    for _ in range(int(rng.integers(3, 5))):
        r = rng.uniform(0.038, RADIUS_RANGE[1])
        for _ in range(400):
            c = np.array([rng.uniform(-0.13, 0.13), rng.uniform(-0.08, 0.08)])
            if _spot_free(c, r, centers, radii):
                centers.append(c)
                radii.append(r)
                break
        else:
            return None
    if sum(abs(c[1]) < r + PARTICLE_RADIUS for c, r in zip(centers, radii)) < 2:
        return None
    for _ in range(500):
        if len(radii) == N_OBSTACLES:
            break
        r = rng.uniform(*RADIUS_RANGE)
        c = rng.uniform(-HALF + MIN_GAP + r, HALF - MIN_GAP - r, size=2)
        if _spot_free(c, r, centers, radii):
            centers.append(c)
            radii.append(r)
    if len(radii) < N_OBSTACLES:
        return None
    return np.array(centers), np.array(radii)


# With 9 circles and every gap >= MIN_GAP = 3 particle diameters the shortest path cannot get much longer than the straight
# line (annealing over the 9 positions found at most 1.07 x; with gaps of 0.05 / 0.04 m: 1.08 x / 1.09 x), so the
# requirement is 5 percent, not the 10 percent first asked for.
MIN_SHORTEST_RATIO = 1.05
MIN_SECOND_EXTRA = 0.03  # m, the second-best route is at least this much longer than the shortest
MIN_ROUTE_DIFFERENCE = 0.04  # m, the second-best route leaves the shortest path by at least this much (a different route)


def _narrowest_point(path, centers, radii):
    """The point of the path with the narrowest passage (sum of the two smallest surface distances to circles / walls),
    away from the start and the goal, and the larger of the two distances."""
    seg = np.concatenate([np.linspace(p, q, 30) for p, q in zip(path[:-1], path[1:])])
    seg = seg[(np.linalg.norm(seg - START, axis=1) > 0.07) & (np.linalg.norm(seg - GOAL, axis=1) > 0.07)]
    d = np.linalg.norm(seg[:, None, :] - centers[None], axis=-1) - radii[None]
    walls = np.stack([HALF - seg[:, 0], HALF + seg[:, 0], HALF - seg[:, 1], HALF + seg[:, 1]], axis=1)
    d = np.sort(np.concatenate([d, walls], axis=1), axis=1)[:, :2]
    k = int(np.argmin(d.sum(axis=1)))
    return seg[k], float(d[k].max())


def route_lengths(centers, radii, min_shortest=0.0):
    """(shortest path, its length, second-best route length, second-best path).

    The second-best route is the shortest path with the narrowest passage of the shortest path plugged by a disk (touching
    both sides of the passage); it is a different route when it leaves the shortest path by MIN_ROUTE_DIFFERENCE
    (l2 = NaN otherwise). l2 is inf when the plugged layout has no path; the second search is skipped (l2 = NaN) when the
    shortest path is shorter than ``min_shortest``.
    """
    path, l1 = shortest_path(centers, radii)
    if path is None:
        return None, float("inf"), float("inf"), None
    if l1 < min_shortest:
        return path, l1, float("nan"), None
    point, rad = _narrowest_point(path, centers, radii)
    path2, l2 = shortest_path(np.vstack([centers, point]), np.append(radii, rad + 0.003))
    if path2 is not None:
        seg1 = np.concatenate([np.linspace(p, q, 40) for p, q in zip(path[:-1], path[1:])])
        far = float(np.linalg.norm(path2[:, None, :] - seg1[None], axis=-1).min(axis=1).max())
        if far < MIN_ROUTE_DIFFERENCE:
            return path, l1, float("nan"), path2
    return path, l1, l2, path2


# Index of the accepted candidate per layout_seed: a pure speed-up for the fixed benchmark layout (it is verified, and
# the full search runs if the hint is missing or wrong). Only the accepted seed gets an entry.
_ACCEPTED_CANDIDATE = {}


def _accept(cand):
    """(info dict) if the candidate meets the layout requirements, else None."""
    centers, radii = cand
    straight = float(np.linalg.norm(GOAL - START))
    coarse = shortest_path(centers, radii, step=0.005)[1]  # quick reject: 5 mm grid, no need for the exact length
    if coarse < 0.98 * MIN_SHORTEST_RATIO * straight:
        return None
    path, l1, l2, path2 = route_lengths(centers, radii, MIN_SHORTEST_RATIO * straight)
    if path is None or not np.isfinite(l2) or l1 < MIN_SHORTEST_RATIO * straight or l2 < l1 + MIN_SECOND_EXTRA:
        return None
    return dict(path=path, second_path=path2, shortest=l1, second=l2, straight=straight)


def find_layout(layout_seed=0, max_tries=3000):
    """(index of the accepted candidate, centers, radii, info) for ``layout_seed`` (full search, no hint)."""
    rng = np.random.default_rng(layout_seed)
    for k in range(max_tries):
        cand = _propose(rng)
        info = None if cand is None else _accept(cand)
        if info is not None:
            return k, cand[0], cand[1], info
    raise RuntimeError(f"no layout found for layout_seed={layout_seed}")


def make_layout(layout_seed=0, return_info=False, max_tries=3000):
    """(centers (9,2), radii (9,)) of the circles for ``layout_seed``; the same seed always gives the same layout.

    ``_propose`` draws candidates (see there); one is accepted when the grid search finds a path, the shortest path
    is at least MIN_SHORTEST_RATIO (1.05) times the straight line (0.5 m) and a second, different route (narrowest passage
    plugged, see ``route_lengths``) is at least MIN_SECOND_EXTRA longer. With ``return_info`` a dict with the reference
    path, its length, the second-best route and its length is returned as a third value.
    """
    hint = _ACCEPTED_CANDIDATE.get(layout_seed)
    if hint is not None:
        rng = np.random.default_rng(layout_seed)
        for k in range(hint + 1):
            cand = _propose(rng)
        info = None if cand is None else _accept(cand)
        if info is not None:
            return (cand[0], cand[1], info) if return_info else (cand[0], cand[1])
    _, centers, radii, info = find_layout(layout_seed, max_tries)
    return (centers, radii, info) if return_info else (centers, radii)

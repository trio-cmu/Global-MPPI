"""Obstacle layout of the obstacle_field task, and the grid-search reference path.

numpy only (no jax / mujoco), so that script/obstacle_field_metrics.py can load it without starting the GPU stack.

The arena is the particle model's square [-0.29, 0.29]^2 (the joint range; the inner faces of the outer walls are at
+-0.29). The point mass has radius 0.01. ``make_layout(layout_seed)`` draws 9 circles with radius in [0.025, 0.045]
by rejection sampling, with a free gap of at least ``MIN_GAP`` (0.04 m = 2 particle diameters) between any two
circles, between a circle and the outer walls, and between a circle and the start or the goal. A layout is only
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
MIN_GAP = 0.04  # m, free gap (surface to surface) between circles / walls / start / goal
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
    """One candidate: a barrier across the straight line with a narrow gap and a wide opening, plus scattered circles.

    The barrier stands at x = x_b (about 0): a chain of overlapping circles (solid) from the bottom tip (leaving a wide
    opening of 8 to 11 cm to the bottom wall) up to just below the narrow gap (surface-to-surface width 4 to 5 cm, about
    2 particle diameters) at height y_g; one circle above the gap closes the rest up to the top wall. Both routes are far
    from the straight line, the remaining circles are scattered at random (min gap MIN_GAP). The whole layout is mirrored
    in y at random. Returns (centers, radii, gap_center, gap_width), or None.
    """
    x_b = rng.uniform(-0.04, 0.04)
    y_tip = rng.uniform(-0.21, -0.18)  # bottom edge of the barrier
    y_g = rng.uniform(0.17, 0.20)  # centre of the narrow gap
    gap = rng.uniform(0.04, 0.05)
    r_cap = (HALF - 0.003 - (y_g + gap / 2)) / 2  # cap circle: bottom edge at the gap, top edge 3 mm below the wall
    if not RADIUS_RANGE[0] <= r_cap <= RADIUS_RANGE[1]:
        return None
    cap = np.array([x_b + rng.uniform(-0.005, 0.005), y_g + gap / 2 + r_cap])
    top_edge = y_g - gap / 2  # the chain ends here, the circle below the gap touches it
    r0, r1 = rng.uniform(0.04, RADIUS_RANGE[1], size=2)
    length = top_edge - y_tip
    n = int(np.ceil((length - r0 - r1) / (0.85 * (r0 + r1)))) + 1  # circles overlap by at least 15 percent
    ys = np.linspace(y_tip + r0, top_edge - r1, n)
    rs = np.linspace(r0, r1, n)
    chain = np.stack([x_b + rng.uniform(-0.006, 0.006, size=n), ys], axis=1)
    if n + 1 > N_OBSTACLES - 2:
        return None
    centers = list(chain) + [cap]
    radii = list(rs) + [r_cap]
    for _ in range(300):
        if len(radii) == N_OBSTACLES:
            break
        r = rng.uniform(*RADIUS_RANGE)
        c = rng.uniform(-HALF + MIN_GAP + r, HALF - MIN_GAP - r, size=2)
        if len(radii) < n + 1 + 2:  # the first two scatter circles stand near the wide route (bottom half), to lengthen it
            c[1] = rng.uniform(-HALF + MIN_GAP + r, -0.04)
        if _spot_free(c, r, centers, radii):
            centers.append(c)
            radii.append(r)
    if len(radii) < N_OBSTACLES:
        return None
    centers, radii = np.array(centers), np.array(radii)
    # the chain circles keep MIN_GAP to the walls except at the tip opening; the cap and the chain bottom are by construction
    gap_center = np.array([x_b, y_g])
    if rng.random() < 0.5:  # mirror in y
        centers = centers * np.array([1.0, -1.0])
        gap_center = gap_center * np.array([1.0, -1.0])
    return centers, radii, gap_center, gap


MIN_SHORTEST_RATIO = 1.25  # the shortest path is at least 25 percent longer than the straight line
MIN_SECOND_EXTRA = 0.07  # m, the second-best route is at least this much longer than the shortest


def route_lengths(centers, radii, gap_center, gap_width, min_shortest=0.0):
    """(shortest path, its length, second-best route length, second-best path): the second-best route is the shortest path with the narrow
    gap plugged. l2 is NaN when the shortest path does not use the gap, inf when the plugged one has no path; when the
    shortest path is shorter than ``min_shortest`` the second search is skipped (l2 = NaN)."""
    path, l1 = shortest_path(centers, radii)
    if path is None:
        return None, float("inf"), float("inf"), None
    seg = np.concatenate([np.linspace(p, q, 40) for p, q in zip(path[:-1], path[1:])])
    if l1 < min_shortest or np.linalg.norm(seg - gap_center, axis=1).min() > gap_width / 2 + 0.02:
        return path, l1, float("nan"), None
    plug_c = np.vstack([centers, gap_center])
    plug_r = np.append(radii, gap_width / 2 + 0.003)
    path2, l2 = shortest_path(plug_c, plug_r)
    return path, l1, l2, path2


# make_layout replays the candidates of the search quickly: the index of the accepted candidate per layout_seed (a pure
# speed-up: it is verified, and the full search runs if the hint is missing or wrong)
_ACCEPTED_CANDIDATE = {0: 30}


def _accept(cand):
    """(info dict) if the candidate meets the layout requirements, else None."""
    centers, radii, gap_center, gap_width = cand
    straight = float(np.linalg.norm(GOAL - START))
    path, l1, l2, path2 = route_lengths(centers, radii, gap_center, gap_width, MIN_SHORTEST_RATIO * straight)
    if path is None or not np.isfinite(l2) or l1 < MIN_SHORTEST_RATIO * straight or l2 < l1 + MIN_SECOND_EXTRA:
        return None
    return dict(path=path, second_path=path2, shortest=l1, second=l2, straight=straight, gap_center=gap_center, gap_width=gap_width)


def make_layout(layout_seed=0, return_info=False, max_tries=3000):
    """(centers (9,2), radii (9,)) of the circles for ``layout_seed``; the same seed always gives the same layout.

    ``_propose`` draws candidates (see there); one is accepted when the grid search finds a path, the shortest path
    goes through the narrow gap, is at least MIN_SHORTEST_RATIO times the straight line (0.5 m), and the second-best
    route (narrow gap plugged) is at least MIN_SECOND_EXTRA longer. With ``return_info`` a dict with the reference
    path, its length, the second-best route length, the straight line and the narrow gap is returned as a third value.
    """
    for hint in (_ACCEPTED_CANDIDATE.get(layout_seed), None):
        rng = np.random.default_rng(layout_seed)
        for k in range(max_tries if hint is None else hint + 1):
            cand = _propose(rng)
            if cand is None or (hint is not None and k < hint):
                continue
            info = _accept(cand)
            if info is not None:
                if return_info:
                    return cand[0], cand[1], info
                return cand[0], cand[1]
        if hint is None:
            break
    raise RuntimeError(f"no layout found for layout_seed={layout_seed}")

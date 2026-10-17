"""T outline helpers (numpy only)."""
import math

import numpy as np

# 12 x 12 cm T in the block frame (crossbar 12 x 4 at y 0.02..0.06, stem 4 x 8 at y -0.06..0.02); matches pusht.xml
OUTLINE = np.array([[-.06, .02], [-.02, .02], [-.02, -.06], [.02, -.06], [.02, .02], [.06, .02], [.06, .06], [-.06, .06]])


def t_polygon(q) -> np.ndarray:
    c, s = math.cos(q[2]), math.sin(q[2])
    return OUTLINE @ np.array([[c, -s], [s, c]]).T + np.asarray(q[:2])


def dist_to_poly(p, poly) -> float:
    """0 inside the polygon, else the distance to its boundary."""
    inside, n = False, len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    if inside:
        return 0.0
    best = 1e9
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        u = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + u * ab))))
    return best

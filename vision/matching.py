"""最小代价指派（匈牙利算法）工具。"""

from __future__ import annotations

import numpy as np


def hungarian_square(cost: np.ndarray) -> list[tuple[int, int]]:
    """求解方阵最小代价指派，返回 ``(row, col)`` 配对。"""

    n = cost.shape[0]
    inf = 1e18
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    minv = np.zeros(n + 1, dtype=np.float64)
    used = np.zeros(n + 1, dtype=bool)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv[:] = inf
        used[:] = False
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = []
    for j in range(1, n + 1):
        if p[j] != 0:
            pairs.append((int(p[j] - 1), int(j - 1)))
    return pairs


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """矩形最小代价指派，支持行列数不等，返回 ``(row, col)`` 索引。"""

    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    transpose = cost.shape[0] > cost.shape[1]
    if transpose:
        cost = cost.T
    rows, cols = cost.shape

    size = max(rows, cols)
    padded = np.full((size, size), 1e12, dtype=np.float64)
    padded[:rows, :cols] = cost
    pairs = hungarian_square(padded)

    valid = [(r, c) for r, c in pairs if r < rows and c < cols]
    row = np.array([r for r, _c in valid], dtype=np.int64)
    col = np.array([c for _r, c in valid], dtype=np.int64)
    if transpose:
        return col, row
    return row, col

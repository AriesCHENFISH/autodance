"""对识别出的队形做对称检测与规整化。"""

from __future__ import annotations

import numpy as np

from vision.matching import linear_sum_assignment


def _points(persons: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(p["x"]), float(p["y"])] for p in persons], dtype=np.float64
    )


def _mirror_assignment(
    points: np.ndarray,
    mirror_points: np.ndarray,
) -> tuple[float, np.ndarray]:
    """返回点集到其镜像点集的匈牙利匹配平均距离和 ``mirror_of`` 映射。"""

    count = points.shape[0]
    if count == 0:
        return float("inf"), np.zeros(0, dtype=np.int64)

    cost = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(count):
            cost[i, j] = np.hypot(
                points[i, 0] - mirror_points[j, 0],
                points[i, 1] - mirror_points[j, 1],
            )
    row, col = linear_sum_assignment(cost)
    total = float(sum(cost[r, c] for r, c in zip(row, col)))
    mirror_of = np.full(count, -1, dtype=np.int64)
    for r, c in zip(row, col):
        mirror_of[r] = c
    return total / count, mirror_of


def _symmetry_costs(
    points: np.ndarray,
    grid_width: int,
    grid_height: int,
) -> tuple[float, float]:
    mirror_left = np.stack(
        [grid_width + 1 - points[:, 0], points[:, 1]], axis=1
    )
    mirror_center = np.stack(
        [grid_width + 1 - points[:, 0], grid_height + 1 - points[:, 1]], axis=1
    )
    left_cost, _ = _mirror_assignment(points, mirror_left)
    center_cost, _ = _mirror_assignment(points, mirror_center)
    return left_cost, center_cost


def _regularize(
    points: np.ndarray,
    mirror_of: np.ndarray,
    grid_width: int,
    grid_height: int,
    central: bool,
) -> np.ndarray:
    result = points.copy()
    count = points.shape[0]

    for i in range(count):
        j = int(mirror_of[i])
        if j < 0 or j <= i:
            continue
        if central:
            new_xi = round((points[i, 0] + (grid_width + 1 - points[j, 0])) / 2)
            new_xj = grid_width + 1 - new_xi
            new_yi = round((points[i, 1] + (grid_height + 1 - points[j, 1])) / 2)
            new_yj = grid_height + 1 - new_yi
        else:
            new_xi = round((points[i, 0] + (grid_width + 1 - points[j, 0])) / 2)
            new_xj = grid_width + 1 - new_xi
            new_yi = round((points[i, 1] + points[j, 1]) / 2)
            new_yj = new_yi
        result[i] = [new_xi, new_yi]
        result[j] = [new_xj, new_yj]

    for i in range(count):
        if mirror_of[i] == i:
            if central:
                result[i] = [round((grid_width + 1) / 2), round((grid_height + 1) / 2)]
            else:
                result[i] = [round((grid_width + 1) / 2), points[i, 1]]

    result = np.clip(result, 1, max(grid_width, grid_height))
    result[:, 0] = np.clip(result[:, 0], 1, grid_width)
    result[:, 1] = np.clip(result[:, 1], 1, grid_height)
    return result


def _has_collision(points: np.ndarray) -> bool:
    rounded = np.round(points).astype(int)
    return len(np.unique(rounded, axis=0)) != rounded.shape[0]


def regularize_formation(
    persons: list[dict],
    grid_width: int,
    grid_height: int,
    max_symmetry_error: float = 1.5,
) -> list[dict] | None:
    """对单个队形做对称检测与规整化。

    自动在左右对称与中心对称间选择代价更小者；若均超过
    ``max_symmetry_error`` 或规整后出现格点冲突，则返回 ``None`` 表示
    保持原样。返回的 ``persons`` 中 ID 不变，仅调整 ``x`` / ``y``。
    """

    if not persons:
        return None
    points = _points(persons)
    count = points.shape[0]
    if count < 2:
        return None

    left_cost, center_cost = _symmetry_costs(points, grid_width, grid_height)
    if left_cost <= center_cost:
        symmetry_type = "left_right"
        cost = left_cost
    else:
        symmetry_type = "central"
        cost = center_cost

    if cost >= max_symmetry_error:
        return None

    if symmetry_type == "central":
        mirror = np.stack(
            [grid_width + 1 - points[:, 0], grid_height + 1 - points[:, 1]], axis=1
        )
    else:
        mirror = np.stack(
            [grid_width + 1 - points[:, 0], points[:, 1]], axis=1
        )
    _, mirror_of = _mirror_assignment(points, mirror)

    regularized = _regularize(
        points, mirror_of, grid_width, grid_height, central=(symmetry_type == "central")
    )
    if _has_collision(regularized):
        return None

    result = []
    for index, person in enumerate(persons):
        result.append(
            {
                "id": person["id"],
                "x": int(round(regularized[index, 0])),
                "y": int(round(regularized[index, 1])),
            }
        )
    return result


def regularize_formations(
    formations: list[dict],
    max_symmetry_error: float = 1.5,
) -> list[dict]:
    """对每个队形尝试对称规整化，直接写回 ``persons`` 坐标。"""

    for formation in formations:
        persons = formation.get("persons")
        if not persons:
            continue
        regularized = regularize_formation(
            persons,
            int(formation["grid_width"]),
            int(formation["grid_height"]),
            max_symmetry_error=max_symmetry_error,
        )
        if regularized is not None:
            formation["persons"] = regularized
    return formations

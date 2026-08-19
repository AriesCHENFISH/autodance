"""Phase 4 队形编辑状态与点击画布渲染。"""

from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np

from .svg_generator import FormationValidationError, normalize_formations


CANVAS_WIDTH = 900
CANVAS_HEIGHT = 680
MARGIN_LEFT = 72
MARGIN_RIGHT = 36
MARGIN_TOP = 105
MARGIN_BOTTOM = 64


def selected_index(formations: list[dict], label: str | None) -> int:
    """从稳定下拉标签解析队形索引。"""

    if not formations:
        raise FormationValidationError("尚未加载队形")
    try:
        formation_id = int(str(label).split("·", 1)[0].strip())
    except (TypeError, ValueError) as error:
        raise FormationValidationError("请选择一个有效队形") from error
    index = formation_id - 1
    if not 0 <= index < len(formations):
        raise FormationValidationError("选择的队形不存在")
    return index


def persons_to_rows(formation: dict) -> list[list[int]]:
    return [
        [int(person["id"]), int(person["x"]), int(person["y"])]
        for person in formation["persons"]
    ]


def _table_rows(table: object) -> list:
    if hasattr(table, "values"):
        return table.values.tolist()
    return list(table or [])


def update_formation(
    formations: list[dict],
    index: int,
    name: str,
    time: float,
    person_table: object,
) -> list[dict]:
    """应用名称、时间和人物坐标表，并重新执行完整校验。"""

    updated = deepcopy(formations)
    if not str(name).strip():
        raise FormationValidationError("队形名称不能为空")
    try:
        time_value = float(time)
    except (TypeError, ValueError) as error:
        raise FormationValidationError("队形时间必须是数字") from error
    if not math.isfinite(time_value) or time_value < 0:
        raise FormationValidationError("队形时间必须是非负有限数字")
    persons = []
    for row in _table_rows(person_table):
        if len(row) < 3:
            raise FormationValidationError("人物坐标表必须包含 ID、X、Y")
        persons.append({"id": row[0], "x": row[1], "y": row[2]})
    updated[index]["name"] = str(name).strip()
    updated[index]["time"] = round(time_value, 3)
    updated[index]["persons"] = persons
    updated[index]["edited"] = True
    return normalize_formations(updated)


def duplicate_formation(formations: list[dict], index: int) -> list[dict]:
    """复制当前队形作为新队形，供用户调整位置和时间。"""

    updated = deepcopy(formations)
    copy_item = deepcopy(updated[index])
    copy_item["name"] = f"{copy_item['name']} 副本"
    copy_item["edited"] = True
    updated.insert(index + 1, copy_item)
    return normalize_formations(updated)


def delete_formation(formations: list[dict], index: int) -> list[dict]:
    if len(formations) <= 1:
        raise FormationValidationError("至少保留一个队形")
    updated = deepcopy(formations)
    del updated[index]
    return normalize_formations(updated)


def canvas_to_grid(
    formation: dict,
    canvas_x: float,
    canvas_y: float,
) -> tuple[int, int]:
    """把点击画布坐标吸附到最近的队形格点。"""

    stage_width = CANVAS_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    stage_height = CANVAS_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    relative_x = (float(canvas_x) - MARGIN_LEFT) / stage_width
    relative_y = (float(canvas_y) - MARGIN_TOP) / stage_height
    column = 1 + round(relative_x * max(1, formation["grid_width"] - 1))
    row = 1 + round(relative_y * max(1, formation["grid_height"] - 1))
    return (
        min(max(column, 1), formation["grid_width"]),
        min(max(row, 1), formation["grid_height"]),
    )


def move_person(
    formations: list[dict],
    index: int,
    person_id: int,
    x: int,
    y: int,
) -> list[dict]:
    updated = deepcopy(formations)
    for person in updated[index]["persons"]:
        if int(person["id"]) == int(person_id):
            person["x"] = int(x)
            person["y"] = int(y)
            updated[index]["edited"] = True
            return normalize_formations(updated)
    raise FormationValidationError(f"当前队形中没有人物 ID {person_id}")


def render_editor_canvas(formation: dict, selected_person_id: int | None = None) -> np.ndarray:
    """渲染可点击的 RGB 编辑画布，布局与 SVG 保持一致。"""

    item = normalize_formations([formation])[0]
    image = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (247, 248, 251), np.uint8)
    stage_width = CANVAS_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    stage_height = CANVAS_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    def point(x: int, y: int) -> tuple[int, int]:
        px = MARGIN_LEFT + round((x - 1) / max(1, item["grid_width"] - 1) * stage_width)
        py = MARGIN_TOP + round((y - 1) / max(1, item["grid_height"] - 1) * stage_height)
        return px, py

    cv2.putText(image, item["name"], (MARGIN_LEFT, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (51, 32, 23), 2, cv2.LINE_AA)
    cv2.putText(image, f"Time {item['time']:.3f}s | {item['grid_width']}x{item['grid_height']}", (MARGIN_LEFT, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (118, 97, 86), 1, cv2.LINE_AA)
    for column in range(1, item["grid_width"] + 1):
        start = point(column, 1)
        end = point(column, item["grid_height"])
        cv2.line(image, start, end, (234, 222, 216), 1, cv2.LINE_AA)
    for row in range(1, item["grid_height"] + 1):
        start = point(1, row)
        end = point(item["grid_width"], row)
        cv2.line(image, start, end, (234, 222, 216), 1, cv2.LINE_AA)
    cv2.rectangle(image, point(1, 1), point(item["grid_width"], item["grid_height"]), (77, 51, 38), 2)
    for person in item["persons"]:
        center = point(person["x"], person["y"])
        hue = (int(person["id"]) * 137.508) % 360
        color_hsv = np.uint8([[[round(hue / 2), 180, 210]]])
        color = tuple(int(v) for v in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0])
        radius = 24 if int(person["id"]) == selected_person_id else 19
        cv2.circle(image, center, radius + 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        label = str(person["id"])
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(image, label, (center[0] - size[0] // 2, center[1] + size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return image

"""关键队形 SVG 渲染、校验和导出。"""

from __future__ import annotations

from copy import deepcopy
from html import escape
import json
from pathlib import Path
import zipfile


class FormationValidationError(ValueError):
    """队形 JSON 或编辑数据不满足 Phase 4 契约。"""


def normalize_formations(payload: object) -> list[dict]:
    """校验 Phase 3 输出并补齐 Phase 4 可编辑字段。"""

    if not isinstance(payload, list) or not payload:
        raise FormationValidationError("formations.json 必须是非空数组")
    normalized = []
    for index, source in enumerate(payload, start=1):
        if not isinstance(source, dict):
            raise FormationValidationError(f"第 {index} 个队形不是对象")
        try:
            grid_width = int(source["grid_width"])
            grid_height = int(source["grid_height"])
            time = float(source["time"])
            persons_source = source["persons"]
        except (KeyError, TypeError, ValueError) as error:
            raise FormationValidationError(f"第 {index} 个队形字段不完整") from error
        if not 1 <= grid_width <= 100 or not 1 <= grid_height <= 100:
            raise FormationValidationError(f"第 {index} 个队形网格尺寸无效")
        if not isinstance(persons_source, list) or not persons_source:
            raise FormationValidationError(f"第 {index} 个队形没有人物")

        persons = []
        seen_ids = set()
        for person in persons_source:
            try:
                person_id = int(person["id"])
                x = int(round(float(person["x"])))
                y = int(round(float(person["y"])))
            except (KeyError, TypeError, ValueError) as error:
                raise FormationValidationError(
                    f"第 {index} 个队形包含无效人物坐标"
                ) from error
            if person_id in seen_ids:
                raise FormationValidationError(
                    f"第 {index} 个队形中人物 ID {person_id} 重复"
                )
            if not 1 <= x <= grid_width or not 1 <= y <= grid_height:
                raise FormationValidationError(
                    f"队形 {index} 的人物 {person_id} 超出网格范围"
                )
            seen_ids.add(person_id)
            persons.append({"id": person_id, "x": x, "y": y})

        formation = deepcopy(source)
        formation["formation_id"] = index
        formation["name"] = str(source.get("name") or f"队形 {index}")
        formation["time"] = round(time, 3)
        formation["grid_width"] = grid_width
        formation["grid_height"] = grid_height
        formation["persons"] = sorted(persons, key=lambda item: item["id"])
        normalized.append(formation)
    return normalized


def formation_label(formation: dict) -> str:
    """生成下拉列表使用的稳定标签。"""

    return (
        f"{int(formation['formation_id'])} · {formation['name']} · "
        f"{float(formation['time']):.3f}s"
    )


def _person_color(person_id: int) -> str:
    hue = round((person_id * 137.508) % 360)
    return f"hsl({hue}, 72%, 48%)"


def render_formation_svg(
    formation: dict,
    canvas_width: int = 900,
    canvas_height: int = 680,
) -> str:
    """将单个标准化队形渲染为自包含 SVG。"""

    item = normalize_formations([formation])[0]
    grid_width = item["grid_width"]
    grid_height = item["grid_height"]
    margin_left, margin_right = 72, 36
    margin_top, margin_bottom = 105, 64
    stage_width = canvas_width - margin_left - margin_right
    stage_height = canvas_height - margin_top - margin_bottom

    def px(x: float) -> float:
        return margin_left + (x - 1) / max(1, grid_width - 1) * stage_width

    def py(y: float) -> float:
        return margin_top + (y - 1) / max(1, grid_height - 1) * stage_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" '
        f'height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}" '
        'role="img">',
        '<rect width="100%" height="100%" fill="#f7f8fb"/>',
        f'<text x="{margin_left}" y="42" font-family="sans-serif" '
        f'font-size="28" font-weight="700" fill="#172033">'
        f'{escape(item["name"])}</text>',
        f'<text x="{margin_left}" y="72" font-family="sans-serif" '
        f'font-size="16" fill="#566176">时间 {item["time"]:.3f}s · '
        f'{grid_width}×{grid_height}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{stage_width}" '
        f'height="{stage_height}" rx="8" fill="#ffffff" stroke="#26334d" '
        'stroke-width="2"/>',
    ]
    for column in range(1, grid_width + 1):
        x = px(column)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{margin_top + stage_height}" stroke="#d8deea" stroke-width="1"/>'
        )
    for row in range(1, grid_height + 1):
        y = py(row)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" '
            f'x2="{margin_left + stage_width}" y2="{y:.2f}" '
            'stroke="#d8deea" stroke-width="1"/>'
        )
    parts.extend(
        [
            f'<text x="20" y="{margin_top + 5}" font-family="sans-serif" '
            'font-size="13" fill="#7a8497">后场</text>',
            f'<text x="20" y="{margin_top + stage_height}" font-family="sans-serif" '
            'font-size="13" fill="#7a8497">前场</text>',
        ]
    )
    for person in item["persons"]:
        x, y = px(person["x"]), py(person["y"])
        color = _person_color(person["id"])
        parts.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="19" fill="{color}" '
                'stroke="#ffffff" stroke-width="4"/>',
                f'<text x="{x:.2f}" y="{y + 6:.2f}" text-anchor="middle" '
                'font-family="sans-serif" font-size="16" font-weight="700" '
                f'fill="#ffffff">{person["id"]}</text>',
                f'<text x="{x:.2f}" y="{y + 39:.2f}" text-anchor="middle" '
                'font-family="sans-serif" font-size="12" fill="#3f4b60">'
                f'({person["x"]}, {person["y"]})</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def write_formation_exports(
    formations: list[dict],
    output_directory: Path,
    selected_index: int = 0,
) -> tuple[Path, Path, Path]:
    """保存编辑 JSON、当前 SVG 及全部 SVG ZIP。"""

    normalized = normalize_formations(formations)
    if not 0 <= selected_index < len(normalized):
        raise FormationValidationError("选择的队形不存在")
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "formations_edited.json"
    json_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    svg_paths = []
    for formation in normalized:
        path = output_directory / f"formation_{formation['formation_id']:03d}.svg"
        path.write_text(render_formation_svg(formation), encoding="utf-8")
        svg_paths.append(path)
    zip_path = output_directory / "formation_svgs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in svg_paths:
            archive.write(path, arcname=path.name)
    return json_path, svg_paths[selected_index], zip_path

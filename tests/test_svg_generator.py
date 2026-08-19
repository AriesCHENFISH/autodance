"""Phase 4 SVG 数据契约和导出测试。"""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from visualization import (
    FormationValidationError,
    normalize_formations,
    render_formation_svg,
    write_formation_exports,
)


FORMATION = {
    "formation_id": 7,
    "time": 1.25,
    "confirmed_at": 1.6,
    "frame_id": 30,
    "grid_width": 20,
    "grid_height": 20,
    "persons": [
        {"id": 2, "x": 15, "y": 12},
        {"id": 1, "x": 5, "y": 8},
    ],
}


class SvgGeneratorTests(unittest.TestCase):
    def test_normalize_adds_names_and_sequential_ids(self) -> None:
        result = normalize_formations([FORMATION])
        self.assertEqual(result[0]["formation_id"], 1)
        self.assertEqual(result[0]["name"], "队形 1")
        self.assertEqual([person["id"] for person in result[0]["persons"]], [1, 2])

    def test_duplicate_person_and_out_of_bounds_are_rejected(self) -> None:
        duplicate = {**FORMATION, "persons": [{"id": 1, "x": 2, "y": 2}] * 2}
        with self.assertRaises(FormationValidationError):
            normalize_formations([duplicate])
        outside = {**FORMATION, "persons": [{"id": 1, "x": 21, "y": 2}]}
        with self.assertRaises(FormationValidationError):
            normalize_formations([outside])

    def test_svg_is_self_contained_and_escapes_name(self) -> None:
        svg = render_formation_svg({**FORMATION, "name": "A < B"})
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("A &lt; B", svg)
        self.assertIn(">1</text>", svg)
        self.assertIn(">2</text>", svg)

    def test_export_writes_json_selected_svg_and_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path, svg_path, zip_path = write_formation_exports(
                [FORMATION, {**FORMATION, "time": 3.5}], Path(directory), 1
            )
            self.assertEqual(len(json.loads(json_path.read_text(encoding="utf-8"))), 2)
            self.assertEqual(svg_path.name, "formation_002.svg")
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["formation_001.svg", "formation_002.svg"],
                )


if __name__ == "__main__":
    unittest.main()

"""Phase 4 队形编辑状态测试。"""

import unittest

from visualization import (
    FormationValidationError,
    canvas_to_grid,
    delete_formation,
    duplicate_formation,
    move_person,
    normalize_formations,
    update_formation,
)


SOURCE = normalize_formations([
    {
        "formation_id": 1,
        "time": 1.0,
        "grid_width": 20,
        "grid_height": 20,
        "persons": [{"id": 1, "x": 3, "y": 4}, {"id": 2, "x": 8, "y": 9}],
    }
])


class FormationEditorTests(unittest.TestCase):
    def test_table_update_marks_formation_edited(self) -> None:
        result = update_formation(SOURCE, 0, "开场", 2.25, [[1, 4, 5], [2, 9, 10]])
        self.assertEqual(result[0]["name"], "开场")
        self.assertEqual(result[0]["time"], 2.25)
        self.assertTrue(result[0]["edited"])
        self.assertEqual(result[0]["persons"][0]["x"], 4)

    def test_click_coordinates_snap_and_person_moves(self) -> None:
        self.assertEqual(canvas_to_grid(SOURCE[0], 72, 105), (1, 1))
        self.assertEqual(canvas_to_grid(SOURCE[0], 864, 616), (20, 20))
        result = move_person(SOURCE, 0, 2, 20, 1)
        self.assertEqual(result[0]["persons"][1], {"id": 2, "x": 20, "y": 1})

    def test_duplicate_and_delete_reindex(self) -> None:
        duplicated = duplicate_formation(SOURCE, 0)
        self.assertEqual(len(duplicated), 2)
        self.assertEqual([item["formation_id"] for item in duplicated], [1, 2])
        self.assertIn("副本", duplicated[1]["name"])
        self.assertEqual(len(delete_formation(duplicated, 0)), 1)
        with self.assertRaises(FormationValidationError):
            delete_formation(SOURCE, 0)


if __name__ == "__main__":
    unittest.main()

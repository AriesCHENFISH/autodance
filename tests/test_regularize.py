"""队形对称规整化测试。"""

import unittest

from formation.regularize import regularize_formation, regularize_formations


def make_persons(items):
    return [
        {"id": index, "x": int(x), "y": int(y)}
        for index, (x, y) in enumerate(items, start=1)
    ]


def left_right_symmetric(persons, width):
    coords = {(p["x"], p["y"]) for p in persons}
    return all((width + 1 - p["x"], p["y"]) in coords for p in persons)


def central_symmetric(persons, width, height):
    coords = {(p["x"], p["y"]) for p in persons}
    return all((width + 1 - p["x"], height + 1 - p["y"]) in coords for p in persons)


class RegularizeFormationTests(unittest.TestCase):
    def test_perfect_left_right_stays_symmetric(self) -> None:
        persons = make_persons([(3, 10), (7, 10), (14, 10), (18, 10)])
        result = regularize_formation(persons, 20, 20)
        self.assertIsNotNone(result)
        self.assertTrue(left_right_symmetric(result, 20))
        self.assertEqual([p["id"] for p in result], [1, 2, 3, 4])

    def test_perturbed_left_right_is_regularized(self) -> None:
        persons = make_persons([(3, 10), (8, 10), (14, 10), (17, 10)])
        result = regularize_formation(persons, 20, 20)
        self.assertIsNotNone(result)
        self.assertTrue(left_right_symmetric(result, 20))
        self.assertEqual(sorted(p["id"] for p in result), [1, 2, 3, 4])

    def test_central_symmetry_is_detected(self) -> None:
        persons = make_persons([(3, 4), (18, 17), (7, 8), (14, 13)])
        result = regularize_formation(persons, 20, 20)
        self.assertIsNotNone(result)
        self.assertTrue(central_symmetric(result, 20, 20))
        self.assertEqual(sorted(p["id"] for p in result), [1, 2, 3, 4])

    def test_asymmetric_returns_none(self) -> None:
        persons = make_persons([(2, 5), (3, 7), (4, 9), (5, 6), (6, 8)])
        self.assertIsNone(regularize_formation(persons, 20, 20))

    def test_coordinates_stay_in_bounds(self) -> None:
        persons = make_persons([(2, 3), (8, 4), (13, 4), (19, 3)])
        result = regularize_formation(persons, 20, 20)
        self.assertIsNotNone(result)
        for p in result:
            self.assertGreaterEqual(p["x"], 1)
            self.assertLessEqual(p["x"], 20)
            self.assertGreaterEqual(p["y"], 1)
            self.assertLessEqual(p["y"], 20)

    def test_regularize_formations_writes_back(self) -> None:
        formations = [
            {
                "formation_id": 1,
                "grid_width": 20,
                "grid_height": 20,
                "persons": make_persons([(3, 10), (8, 10), (14, 10), (17, 10)]),
            },
            {
                "formation_id": 2,
                "grid_width": 20,
                "grid_height": 20,
                "persons": make_persons([(2, 5), (3, 7), (4, 9), (5, 6), (6, 8)]),
            },
        ]
        regularize_formations(formations)
        self.assertTrue(left_right_symmetric(formations[0]["persons"], 20))
        # 不对称队形保持原样
        self.assertEqual(formations[1]["persons"], make_persons([(2, 5), (3, 7), (4, 9), (5, 6), (6, 8)]))


if __name__ == "__main__":
    unittest.main()

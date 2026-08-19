"""关键队形状态机测试。"""

import unittest

from formation.analyzer import analyze_formations


FPS = 20.0
BASE = {1: (1.0, 2.0), 2: (3.0, 2.0), 3: (5.0, 6.0), 4: (7.0, 6.0)}


def make_frames(
    positions: list[dict[int, tuple[float, float]]],
    interpolated: set[tuple[int, int]] | None = None,
) -> list[dict]:
    interpolated = interpolated or set()
    return [
        {
            "frame_id": frame_id,
            "timestamp": round(frame_id / FPS, 3),
            "persons": [
                {
                    "id": identity_id,
                    "stage_x": point[0],
                    "stage_y": point[1],
                    "in_stage": True,
                    "interpolated": (frame_id, identity_id) in interpolated,
                }
                for identity_id, point in sorted(frame_positions.items())
            ],
        }
        for frame_id, frame_positions in enumerate(positions)
    ]


class FormationAnalyzerTests(unittest.TestCase):
    def test_static_people_create_only_initial_formation(self) -> None:
        formations = analyze_formations(make_frames([BASE] * 80), FPS)
        self.assertEqual(len(formations), 1)
        self.assertEqual(formations[0]["formation_id"], 1)
        self.assertEqual(len(formations[0]["persons"]), 4)

    def test_only_one_person_moves_does_not_create_formation(self) -> None:
        moved = {**BASE, 1: (4.0, 2.0)}
        formations = analyze_formations(make_frames([BASE] * 25 + [moved] * 50), FPS)
        self.assertEqual(len(formations), 1)

    def test_two_people_move_less_than_two_target_cells(self) -> None:
        moved = {**BASE, 1: (1.8, 2.0), 2: (3.8, 2.0)}
        formations = analyze_formations(make_frames([BASE] * 25 + [moved] * 50), FPS)
        self.assertEqual(len(formations), 1)

    def test_two_people_move_and_stabilize_create_new_formation(self) -> None:
        moved = {**BASE, 1: (3.0, 2.0), 2: (5.0, 2.0)}
        formations = analyze_formations(make_frames([BASE] * 25 + [moved] * 50), FPS)
        self.assertEqual(len(formations), 2)
        self.assertGreaterEqual(formations[1]["time"], 1.5)
        self.assertGreater(formations[1]["confirmed_at"], formations[1]["time"])

    def test_output_contract_is_ready_for_svg_consumer(self) -> None:
        formation = analyze_formations(make_frames([BASE] * 20), FPS)[0]

        self.assertEqual(
            set(formation),
            {
                "formation_id",
                "time",
                "confirmed_at",
                "frame_id",
                "grid_width",
                "grid_height",
                "persons",
            },
        )
        self.assertEqual(formation["formation_id"], 1)
        self.assertLessEqual(formation["time"], formation["confirmed_at"])
        self.assertEqual(formation["grid_width"], 20)
        self.assertEqual(formation["grid_height"], 20)
        self.assertEqual(
            [person["id"] for person in formation["persons"]],
            sorted(BASE),
        )
        for person in formation["persons"]:
            self.assertIsInstance(person["x"], int)
            self.assertIsInstance(person["y"], int)
            self.assertGreaterEqual(person["x"], 1)
            self.assertLessEqual(person["x"], 20)
            self.assertGreaterEqual(person["y"], 1)
            self.assertLessEqual(person["y"], 20)

    def test_motion_without_stability_does_not_create_new_formation(self) -> None:
        positions = [BASE] * 25
        for frame_id in range(50):
            offset = 2.0 if frame_id % 2 == 0 else 4.0
            positions.append({**BASE, 1: (offset, 2.0), 2: (offset + 2.0, 2.0)})
        self.assertEqual(len(analyze_formations(make_frames(positions), FPS)), 1)

    def test_missing_and_interpolated_windows_are_not_stable(self) -> None:
        missing = [dict(BASE) for _ in range(15)]
        for frame_id in range(4):
            missing[frame_id].pop(4)
        self.assertEqual(analyze_formations(make_frames(missing), FPS), [])

        interpolation = {(frame_id, 4) for frame_id in range(5)}
        self.assertEqual(
            analyze_formations(make_frames([BASE] * 15, interpolation), FPS), []
        )

    def test_single_outlier_and_boundary_jitter_do_not_duplicate(self) -> None:
        positions = [BASE] * 20 + [{**BASE, 1: (9.0, 9.0)}] + [BASE] * 40
        self.assertEqual(len(analyze_formations(make_frames(positions), FPS)), 1)

    def test_grid_sizes_control_ranges_and_distance_scale(self) -> None:
        edge = {1: (0.0, 0.0), 2: (9.0, 9.0)}
        for width, height in ((8, 8), (20, 20), (30, 20)):
            formation = analyze_formations(make_frames([edge] * 15), FPS, width, height)[0]
            self.assertEqual(formation["persons"][0]["x"], 1)
            self.assertEqual(formation["persons"][0]["y"], 1)
            self.assertEqual(formation["persons"][1]["x"], width)
            self.assertEqual(formation["persons"][1]["y"], height)

        slightly_moved = {**BASE, 1: (2.0, 2.0), 2: (4.0, 2.0)}
        frames = make_frames([BASE] * 25 + [slightly_moved] * 50)
        self.assertEqual(len(analyze_formations(frames, FPS, 8, 8)), 1)
        self.assertEqual(len(analyze_formations(frames, FPS, 30, 20)), 2)


if __name__ == "__main__":
    unittest.main()

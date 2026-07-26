"""固定人数身份归并测试。"""

import unittest

import numpy as np

from vision.identity import (
    consolidate_track_ids,
    extract_appearance_descriptor,
    relabel_tracked_frames,
)
from vision.tracker import TrackedPerson


def _person(source_id: int, x: int) -> TrackedPerson:
    return TrackedPerson(
        person_id=source_id,
        x=float(x),
        y=100.0,
        box=(x - 10, 20, x + 10, 100),
    )


class IdentityConsolidationTests(unittest.TestCase):
    def test_fragmented_track_is_merged_by_appearance_and_motion(self) -> None:
        frames = []
        for frame_id in range(10):
            first_id = 1 if frame_id < 5 else 3
            frames.append([_person(first_id, 20 + frame_id), _person(2, 120)])

        descriptors = {
            1: [np.array([1.0, 0.0], dtype=np.float32)],
            2: [np.array([0.0, 1.0], dtype=np.float32)],
            3: [np.array([1.0, 0.0], dtype=np.float32)],
        }
        result = consolidate_track_ids(frames, descriptors, expected_count=2)

        self.assertEqual(result.final_id_count, 2)
        self.assertEqual(result.mapping[1], result.mapping[3])
        self.assertNotEqual(result.mapping[1], result.mapping[2])

        relabeled = relabel_tracked_frames(frames, result)
        self.assertTrue(all({person.person_id for person in frame} == {1, 2} for frame in relabeled))

    def test_fully_overlapping_extra_track_is_dropped(self) -> None:
        frames = [
            [_person(1, 20), _person(2, 120), _person(9, 220)]
            for _ in range(6)
        ]
        descriptors = {
            1: [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
            2: [np.array([0.0, 1.0, 0.0], dtype=np.float32)],
            9: [np.array([0.0, 0.0, 1.0], dtype=np.float32)],
        }

        result = consolidate_track_ids(frames, descriptors, expected_count=2)

        self.assertEqual(result.final_id_count, 2)
        self.assertEqual(len(result.dropped_ids), 1)
        relabeled = relabel_tracked_frames(frames, result)
        self.assertTrue(all(len(frame) == 2 for frame in relabeled))

    def test_appearance_descriptor_is_normalized(self) -> None:
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        frame[10:110, 30:90] = (20, 80, 200)

        descriptor = extract_appearance_descriptor(frame, (30, 10, 90, 110))

        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

"""逐帧固定身份分配测试。"""

import unittest

import numpy as np

from vision.identity import assign_fixed_identities
from vision.tracker import TrackedPerson


def _person(source_id: int, x: float) -> TrackedPerson:
    """创建测试人物。"""

    return TrackedPerson(source_id, x, 100.0, (round(x - 10), 20, round(x + 10), 100), 0.9)


class FixedIdentityTests(unittest.TestCase):
    def test_online_id_swap_is_corrected_by_appearance(self) -> None:
        """在线 ID 交换后，最终身份仍应跟随衣着特征。"""

        frames = [
            [_person(1, 20 + index), _person(2, 120 - index)]
            if index < 3
            else [_person(1, 120 - index), _person(2, 20 + index)]
            for index in range(6)
        ]
        first = np.array([1.0, 0.0], dtype=np.float32)
        second = np.array([0.0, 1.0], dtype=np.float32)
        descriptors = [
            {1: first, 2: second} if index < 3 else {1: second, 2: first}
            for index in range(6)
        ]

        relabeled, result = assign_fixed_identities(
            frames, descriptors, 2, 200, 200
        )

        self.assertEqual(result.final_id_count, 2)
        self.assertLess(relabeled[-1][0].x, relabeled[-1][1].x)
        self.assertEqual(
            {person.person_id for person in relabeled[-1]}, {1, 2}
        )


if __name__ == "__main__":
    unittest.main()

"""视频视觉处理模块。"""

from .detector import PersonDetector
from .identity import (
    IdentityConsolidation,
    consolidate_track_ids,
    extract_appearance_descriptor,
    relabel_tracked_frames,
)
from .tracker import TrackedPerson, draw_tracked_persons, extract_tracked_persons

__all__ = [
    "consolidate_track_ids",
    "extract_appearance_descriptor",
    "IdentityConsolidation",
    "PersonDetector",
    "relabel_tracked_frames",
    "TrackedPerson",
    "draw_tracked_persons",
    "extract_tracked_persons",
]

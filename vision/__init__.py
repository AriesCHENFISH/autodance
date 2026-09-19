"""视频视觉处理模块。"""

from .auto_calibration import auto_calibrate_stage
from .coordinate import CalibrationError, StageCalibration, StagePoint
from .detector import PersonDetector
from .identity import FixedIdentityAssignment, assign_fixed_identities
from .identity import (
    IdentityConsolidation,
    KeyframeIdentityAssignment,
    assign_keyframe_identities,
    consolidate_track_ids,
    extract_appearance_descriptor,
    relabel_tracked_frames,
)
from .matching import linear_sum_assignment
from .tracker import (
    TrackedPerson,
    draw_tracked_persons,
    extract_detected_persons,
    extract_tracked_persons,
)
__all__ = [
    "auto_calibrate_stage",
    "CalibrationError",
    "consolidate_track_ids",
    "extract_appearance_descriptor",
    "IdentityConsolidation",
    "PersonDetector",
    "FixedIdentityAssignment",
    "KeyframeIdentityAssignment",
    "assign_fixed_identities",
    "assign_keyframe_identities",
    "linear_sum_assignment",
    "relabel_tracked_frames",
    "StageCalibration",
    "StagePoint",
    "TrackedPerson",
    "draw_tracked_persons",
    "extract_detected_persons",
    "extract_tracked_persons",
]

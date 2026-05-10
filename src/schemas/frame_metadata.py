from enum import IntFlag
from pydantic import BaseModel

class TransportFlags(IntFlag):
    NONE = 0
    FORCE_ANCHOR = 1
    LOW_CONFIDENCE = 2
    OCCLUSION_RISK = 4

class FrameMetadata(BaseModel):
    frame_idx: int
    presentation_ts: float               
    monotonic_timestamp: int             
    wall_clock_arrival_ts: int           
    scene_epoch: int = 0                 
    drift_score: float = 0.0             
    transport_flags: TransportFlags = TransportFlags.NONE
    is_anchor: bool = False
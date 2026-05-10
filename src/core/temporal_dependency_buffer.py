import time
import torch
from typing import List, Optional
from src.schemas.frame_metadata import FrameMetadata, TransportFlags

class TensorPool:
    def __init__(self, capacity: int, height: int, width: int, device: str = 'cuda'):
        self.capacity = capacity
        self.height = height
        self.width = width
        self.device = device
        
        self.conf_h = height // 4
        self.conf_w = width // 4
        
        self._rgb_pool = torch.zeros((capacity, 3, height, width), device=device, dtype=torch.float16)
        self._alpha_pool = torch.zeros((capacity, 1, height, width), device=device, dtype=torch.float16)
        self._confidence_pool = torch.zeros((capacity, 1, self.conf_h, self.conf_w), device=device, dtype=torch.float16)
        
        self._free_indices = list(range(capacity))

    def acquire(self) -> int:
        if not self._free_indices: raise MemoryError("TensorPool kapasitesi doldu!")
        return self._free_indices.pop(0)

    def release(self, idx: int):
        self._free_indices.append(idx)

class FrameTensorBundle:
    __slots__ = ['pool_idx', 'rgb_nchw', 'alpha_core', 'confidence_map']
    def __init__(self, pool: TensorPool, pool_idx: int):
        self.pool_idx = pool_idx
        self.rgb_nchw = pool._rgb_pool[pool_idx]
        self.alpha_core = pool._alpha_pool[pool_idx]
        self.confidence_map = pool._confidence_pool[pool_idx]

class FrameState:
    __slots__ = ['metadata', 'tensors']
    def __init__(self, metadata: FrameMetadata, tensors: FrameTensorBundle):
        self.metadata = metadata
        self.tensors = tensors

class CausalRingBuffer:
    def __init__(self, width: int, height: int, capacity: int = 16, active_window: int = 5, device: str = 'cuda'):
        self.capacity = capacity
        self.active_window = active_window
        self.pool = TensorPool(capacity=capacity, height=height, width=width, device=device)
        self.buffer: List[Optional[FrameState]] = [None] * capacity
        self.head_idx = -1
        self._count = 0

    def push(self, frame_idx: int, pts: float, rgb_tensor: torch.Tensor, alpha_tensor: torch.Tensor, is_anchor: bool = False):
        next_idx = (self.head_idx + 1) % self.capacity
        if self.buffer[next_idx] is not None:
            self.pool.release(self.buffer[next_idx].tensors.pool_idx)
            self.buffer[next_idx] = None
            self._count -= 1

        pool_idx = self.pool.acquire()
        tensors = FrameTensorBundle(self.pool, pool_idx)
        
        tensors.rgb_nchw.copy_(rgb_tensor)
        tensors.alpha_core.copy_(alpha_tensor)

        metadata = FrameMetadata(
            frame_idx=frame_idx, presentation_ts=pts,
            monotonic_timestamp=time.perf_counter_ns(),
            wall_clock_arrival_ts=int(time.time() * 1e9), is_anchor=is_anchor
        )
        self.buffer[next_idx] = FrameState(metadata, tensors)
        self.head_idx = next_idx
        self._count = min(self.capacity, self._count + 1)

    def get_causal_window(self) -> List[FrameState]:
        if self._count == 0: return []
        window_size = min(self.active_window, self._count)
        return [self.buffer[(self.head_idx - step) % self.capacity] for step in range(window_size - 1, -1, -1)]
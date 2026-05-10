import cv2
import numpy as np
from src.core.temporal_dependency_buffer import FrameState

class SemanticPacketizer:
    def __init__(self, quality: int = 80):
        self.quality = quality

    def simulate_payload(self, current_state: FrameState) -> dict:
        # Uyarı: Bu modül gerçek HW Encoder (NVENC) yazılana kadar CPU'ya veri çeker ve darboğaz yaratır.
        rgb = (current_state.tensors.rgb_nchw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        alpha = current_state.tensors.alpha_core[0].cpu().numpy()
        conf = current_state.tensors.confidence_map[0].cpu().numpy()

        _, full_encoded = cv2.imencode('.webp', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, self.quality])
        full_size_kb = len(full_encoded) / 1024.0

        mask_3c = np.expand_dims(alpha > 0.05, axis=-1)
        foreground_rgb = np.where(mask_3c, rgb, 0).astype(np.uint8)
        _, fg_encoded = cv2.imencode('.webp', cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, self.quality])

        h, w = alpha.shape
        alpha_down = cv2.resize((alpha * 255).astype(np.uint8), (w//2, h//2))
        conf_down = cv2.resize((conf * 255).astype(np.uint8), (w//2, h//2))

        _, alpha_encoded = cv2.imencode('.webp', alpha_down, [cv2.IMWRITE_WEBP_QUALITY, 50])
        _, conf_encoded = cv2.imencode('.webp', conf_down, [cv2.IMWRITE_WEBP_QUALITY, 50])

        semantic_size_kb = (len(fg_encoded) + len(alpha_encoded) + len(conf_encoded)) / 1024.0
        return {"full_size_kb": full_size_kb, "semantic_size_kb": semantic_size_kb, "savings_percent": 100 * (1.0 - (semantic_size_kb / full_size_kb))}
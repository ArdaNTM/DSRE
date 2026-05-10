import torch
import torch.nn.functional as F
from typing import List
from src.core.temporal_dependency_buffer import FrameState

class TemporalConfidenceEngine:
    """
    Faz 2.1: Tamamen GPU-Resident, Fused Tensor Operatörleri ile çalışan Güven Motoru.
    """
    def __init__(self, window_size: int = 5, base_beta: float = 0.5, decay_k: float = 20.0, device: str = 'cuda'):
        self.window_size = window_size
        self.base_beta = base_beta
        self.decay_k = decay_k
        self.device = device
        
        # [REVIEW FIX]: Sözlük (Dict) Anti-Pattern'i kaldırıldı, doğrudan attribute yapıldı.
        scharr_kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        scharr_kernel_y = scharr_kernel_x.transpose(2, 3)
        self.sx = scharr_kernel_x
        self.sy = scharr_kernel_y

    @torch.no_grad()
    def compute_confidence(self, causal_window: List[FrameState]):
        if len(causal_window) < 2:
            causal_window[-1].tensors.confidence_map.fill_(1.0)
            return

        current_state = causal_window[-1]
        previous_state = causal_window[-2]
        
        # 1. GPU üzerinde donanımsal Interpolate (1/4 Downsample)
        alpha_stack = torch.stack([state.tensors.alpha_core for state in causal_window]).to(torch.float32)
        alpha_down = F.interpolate(alpha_stack, scale_factor=0.25, mode='area') 
        
        current_alpha_down = alpha_down[-1:] 
        
        # 2. Gradient-Aware Spatial Confidence (GPU Conv2D ile)
        blurred_alpha = F.avg_pool2d(current_alpha_down, kernel_size=3, stride=1, padding=1)
        
        gx = F.conv2d(blurred_alpha, self.sx, padding=1)
        gy = F.conv2d(blurred_alpha, self.sy, padding=1)
        
        # [REVIEW FIX]: Sqrt(0) kaynaklı sonsuzluk (NaN) riskini önlemek için epsilon eklendi
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        alpha_penalty = 1.0 - torch.exp(-10.0 * (current_alpha_down - 0.5)**2)
        edge_consistency = 1.0 - torch.exp(-5.0 * grad_mag)
        spatial_conf = alpha_penalty * edge_consistency

        # 3. Temporal Confidence (VRAM üzerinde Variance)
        # [REVIEW FIX]: NumPy'daki (ddof=0) ile uyumlu çalışması için unbiased=False eklendi
        temporal_var = torch.var(alpha_down, dim=0, keepdim=True, unbiased=False)
        temporal_conf = torch.exp(-self.decay_k * temporal_var)

        # 4. Adaptive Local Fusion (Context-Adaptive)
        local_var = F.avg_pool2d(temporal_var, kernel_size=15, stride=1, padding=7)
        w_t = torch.clamp(1.0 - (local_var * 5.0), 0.2, 0.8)
        w_s = 1.0 - w_t
        
        raw_conf = (w_t * temporal_conf) + (w_s * spatial_conf)

        # 5. EMA & Tensöre Yazım
        prev_conf = previous_state.tensors.confidence_map.to(torch.float32)
        smoothed_conf = (self.base_beta * raw_conf) + ((1.0 - self.base_beta) * prev_conf)
        
        # [REVIEW FIX]: GPU'nun eklediği 4. boyutu (Batch) havuza yazmadan önce Squeeze ile atıyoruz
        current_state.tensors.confidence_map.copy_(smoothed_conf.squeeze(0).to(torch.float16))
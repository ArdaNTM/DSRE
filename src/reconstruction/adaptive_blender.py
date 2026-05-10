import torch
import torch.nn.functional as F
import contextlib
from src.core.temporal_dependency_buffer import FrameState

class AdaptiveBlender:
    def __init__(self, blur_k: int = 15, device: str = 'cuda'):
        self.blur_k = blur_k
        self.device = device
        
        self.kx = torch.tensor([[[-1, 0, 1]]], device=device, dtype=torch.float32).view(1,1,1,3)
        self.ky = torch.tensor([[[-1], [0], [1]]], device=device, dtype=torch.float32).view(1,1,3,1)
        
        gauss_1d = torch.tensor([1., 4., 6., 4., 1.], device=device, dtype=torch.float32)
        gauss_1d /= gauss_1d.sum()
        self.gauss_h = gauss_1d.view(1, 1, 1, 5)
        self.gauss_v = gauss_1d.view(1, 1, 5, 1)

    @torch.no_grad()
    def blend(self, current_state: FrameState, clean_plate_tensor: torch.Tensor) -> torch.Tensor:
        if 'cuda' in self.device:
            autocast_ctx = torch.autocast('cuda', dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            fg_rgb = current_state.tensors.rgb_nchw
            alpha_raw = current_state.tensors.alpha_core
            conf_low = current_state.tensors.confidence_map

            conf_full = F.interpolate(conf_low.unsqueeze(0), scale_factor=4.0, mode='bilinear', align_corners=False)
            
            blurred_conf = F.conv2d(conf_full, self.gauss_h, padding=(0,2))
            blurred_conf = F.conv2d(blurred_conf, self.gauss_v, padding=(2,0)).squeeze(0)
            
            alpha_soft = F.conv2d(alpha_raw.unsqueeze(0), self.gauss_h, padding=(0,2))
            alpha_soft = F.conv2d(alpha_soft, self.gauss_v, padding=(2,0)).squeeze(0)
            
        with torch.autocast(device_type=self.device, enabled=False) if 'cuda' in self.device else contextlib.nullcontext():
            alpha_fp32 = alpha_raw.unsqueeze(0).float()
            grad_x = F.conv2d(alpha_fp32, self.kx, padding=(0,1))
            grad_y = F.conv2d(alpha_fp32, self.ky, padding=(1,0))
            
            grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-4).squeeze(0)
            edge_weight = 1.0 - torch.exp(-5.0 * grad_mag)
            
        with autocast_ctx:
            edge_weight = edge_weight.to(conf_full.dtype)
            
            conf_full_sq = conf_full.squeeze(0)
            conf_full_blend = (conf_full_sq * edge_weight) + (blurred_conf * (1.0 - edge_weight))
            final_alpha = (alpha_raw * conf_full_blend) + (alpha_soft * (1.0 - conf_full_blend))
            composited = (fg_rgb * final_alpha) + (clean_plate_tensor * (1.0 - final_alpha))
            
            return composited.clamp(0.0, 1.0)
import torch
import torch.nn.functional as F

class ObjectiveMetricsEngine:
    """
    Faz 4.1: Motion-Compensated Temporal Evaluation & True Trimap
    (FFT Faz Korelasyonu ile Kamera Hareketi Telafisi İçerir)
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.psnr_history = []
        self.mse_history = []
        self.mc_energy_history = []
        
        self.prev_recon = None 
        self.prev_gray = None
        
        self.dilation_kernel = torch.ones((1, 1, 5, 5), device=device, dtype=torch.float32)

    def _compute_phase_correlation(self, curr_gray: torch.Tensor, prev_gray: torch.Tensor):
        # FAZ 4.1 HOTFIX: cuFFT FP16'da 2'nin üssü boyut dayatması yaptığı için 
        # işlemi donanımsal olarak güvenli FP32 uzayında yapıyoruz.
        curr_gray_fp32 = curr_gray.float()
        prev_gray_fp32 = prev_gray.float()
        
        # FFT tabanlı 2D Faz Korelasyonu (Sub-pixel kamera hareketini bulur)
        f_curr = torch.fft.fft2(curr_gray_fp32)
        f_prev = torch.fft.fft2(prev_gray_fp32)
        
        cross_power = f_prev * torch.conj(f_curr)
        cross_power = cross_power / (torch.abs(cross_power) + 1e-8)
        corr = torch.fft.ifft2(cross_power).real
        
        # FAZ 4.1 HOTFIX 2: Tensor 3 Boyutlu (C, H, W) geldiği için B'yi (Batch) çıkardık.
        C, H, W = corr.shape
        max_idx = torch.argmax(corr.view(-1))
        dy = (max_idx // W).item()
        dx = (max_idx % W).item()
        
        if dy > H // 2: dy -= H
        if dx > W // 2: dx -= W
        
        return dy, dx

    def _apply_translation(self, tensor: torch.Tensor, dy: int, dx: int):
        if dy == 0 and dx == 0:
            return tensor
            
        C, H, W = tensor.shape
        norm_dx = -2.0 * dx / W 
        norm_dy = -2.0 * dy / H
        
        theta = torch.tensor([[[1.0, 0.0, norm_dx], [0.0, 1.0, norm_dy]]], device=self.device, dtype=torch.float32)
        grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)
        shifted = F.grid_sample(tensor.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return shifted.squeeze(0)

    @torch.no_grad()
    def accumulate(self, original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor, alpha_mask: torch.Tensor, curr_gray: torch.Tensor):
        # 1. PSEUDO-TRIMAP EVALUATION
        unknown = ((alpha_mask.float() > 0.05) & (alpha_mask.float() < 0.95)).float()
        dilated = F.conv2d(unknown.unsqueeze(0), self.dilation_kernel, padding=2) > 0
        boundary_mask = dilated.squeeze(0) 
        
        boundary_mask_3c = boundary_mask.expand_as(original_tensor) 
        mask_f = boundary_mask_3c.float()
        
        valid_pixels = torch.clamp(mask_f.sum(), min=1.0)
        
        diff = reconstructed_tensor.float() - original_tensor.float()
        diff_sq = torch.square(diff) 
        masked_diff_sq = diff_sq * mask_f.to(diff_sq.dtype)
        
        mse = masked_diff_sq.sum() / valid_pixels
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-9))

        self.mse_history.append(mse.item())
        self.psnr_history.append(psnr.item())

        # 2. MOTION-COMPENSATED FRAME DIFFERENCE ENERGY (Gerçek Flicker Metriği)
        if self.prev_recon is not None and self.prev_gray is not None:
            dy, dx = self._compute_phase_correlation(curr_gray, self.prev_gray)
            
            # Kameranın fiziksel hareketini bir önceki kareden çıkararak telafi ediyoruz (Motion Compensation)
            mc_prev_recon = self._apply_translation(self.prev_recon, dy, dx)
            
            # Kalan fark = Saf AI Titremesi (Flicker)
            energy = torch.mean(torch.abs(reconstructed_tensor.float() - mc_prev_recon.float())).item()
            self.mc_energy_history.append(energy)
        else:
            self.prev_recon = torch.empty_like(reconstructed_tensor)
            self.prev_gray = torch.empty_like(curr_gray)
            
        self.prev_recon.copy_(reconstructed_tensor)
        self.prev_gray.copy_(curr_gray)

    def get_report(self) -> dict:
        if not self.psnr_history:
            return {"avg_psnr": 0.0, "avg_mse": 0.0, "avg_mc_energy": 0.0, "frames_evaluated": 0}

        avg_psnr = sum(self.psnr_history) / len(self.psnr_history)
        avg_mse = sum(self.mse_history) / len(self.mse_history)
        avg_mc_energy = sum(self.mc_energy_history) / len(self.mc_energy_history) if self.mc_energy_history else 0.0

        return {
            "avg_psnr": avg_psnr,
            "avg_mse": avg_mse,
            "avg_mc_energy": avg_mc_energy,
            "frames_evaluated": len(self.psnr_history)
        }
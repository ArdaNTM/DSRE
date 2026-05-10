import torch

class TriBandEntropyCodec:
    """
    Faz 5: Tri-Band (RGB, Alpha, Confidence) Entropi Kodlayıcısı ve Olasılık Modelleyicisi.
    Tensörleri 8-bit (0-255) uzayında sembollere dönüştürür, PDF/CDF (Olasılık Yoğunluk Fonksiyonu) 
    çıkarır ve Shannon Entropisi formülüyle Aritmetik Kodlama'nın ulaşacağı gerçek bit boyutunu hesaplar.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.total_payload_bytes = 0.0
        self.frame_bpp_history = []

    @torch.no_grad()
    def _compute_shannon_entropy(self, tensor_8bit: torch.Tensor) -> float:
        # Tensörü tek boyutlu (flatten) hale getir
        flat_tensor = tensor_8bit.flatten()
        
        # Olasılık Dağılımını (PDF) çıkar (0-255 arası)
        # torch.bincount, GPU üzerinde histogramdan çok daha hızlı ve deterministiktir
        counts = torch.bincount(flat_tensor, minlength=256).float()
        
        # Sadece var olan sembolleri al (log2'de 0'a bölme hatasını önlemek için)
        probabilities = counts[counts > 0] / flat_tensor.numel()
        
        # Shannon Entropisi Formülü: H(X) = - sum( p(x) * log2(p(x)) )
        # Çıkan sonuç, bir pikseli ifade etmek için gereken teorik minimum bit sayısıdır
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
        return entropy.item()

    @torch.no_grad()
    def encode_frame(self, rgb_residual: torch.Tensor, alpha_mask: torch.Tensor, conf_map: torch.Tensor):
        """
        Gelen tensörleri nicemler (quantize), entropilerini hesaplar ve bayt boyutunu döndürür.
        """
        # 1. Nicemleme (Quantization) FP32 -> 8-bit Integer (0-255)
        # Ağ üzerinden FP32 gönderilmez, sembollere (symbols) dönüştürülmelidir.
        rgb_8bit = (rgb_residual.clamp(0, 1) * 255.0).to(torch.uint8)
        alpha_8bit = (alpha_mask.clamp(0, 1) * 255.0).to(torch.uint8)
        conf_8bit = (conf_map.clamp(0, 1) * 255.0).to(torch.uint8)

        # 2. Shannon Entropisi Hesaplama (Bits Per Symbol)
        h_rgb = self._compute_shannon_entropy(rgb_8bit)
        h_alpha = self._compute_shannon_entropy(alpha_8bit)
        h_conf = self._compute_shannon_entropy(conf_8bit)

        # 3. Gerçek Bit ve Bayt Yükü Hesaplama
        # RGB 3 kanal, Alpha 1 kanal, Conf ise 1/4 çözünürlükte (W/4, H/4) 1 kanal
        rgb_bits = h_rgb * rgb_8bit.numel()
        alpha_bits = h_alpha * alpha_8bit.numel()
        conf_bits = h_conf * conf_8bit.numel()

        total_bits = rgb_bits + alpha_bits + conf_bits
        frame_bytes = total_bits / 8.0
        
        # Aritmetik kodlama (Arithmetic Coder) header ve state overhead'i (Yaklaşık %2)
        frame_bytes = frame_bytes * 1.02 

        # İstatistik Kaydı (BPP - Bits Per Pixel)
        # Toplam biti, orijinal ekranın piksel sayısına (W*H) bölüyoruz.
        pixels = alpha_mask.numel()
        bpp = total_bits / pixels

        self.total_payload_bytes += frame_bytes
        self.frame_bpp_history.append(bpp)

        return frame_bytes / 1024.0 # KB cinsinden döndür

    def get_report(self) -> dict:
        avg_bpp = sum(self.frame_bpp_history) / len(self.frame_bpp_history) if self.frame_bpp_history else 0.0
        return {
            "total_payload_kb": self.total_payload_bytes / 1024.0,
            "total_payload_mb": self.total_payload_bytes / (1024.0 * 1024.0),
            "avg_bpp": avg_bpp
        }
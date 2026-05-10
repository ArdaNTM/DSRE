import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F

from src.external.rvm_wrapper import RVMWrapper
from src.core.temporal_dependency_buffer import CausalRingBuffer
from src.core.temporal_confidence import TemporalConfidenceEngine
from src.reconstruction.adaptive_blender import AdaptiveBlender
from src.core.metrics_engine import ObjectiveMetricsEngine
from src.transport.entropy_codec import TriBandEntropyCodec  # FAZ 5 YENİ EKLENDİ

def run_real_video_test(video_path: str, output_path: str):
    print("\n[DSRE] FAZ 5 BAŞLIYOR... (TRI-BAND ENTROPY CODING & BITSTREAM MODELING)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    fps = cap.get(cv2.CAP_PROP_FPS)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rvm = RVMWrapper(device=device)
    
    buffer = CausalRingBuffer(width=orig_width, height=orig_height, capacity=16, active_window=5, device=device)
    confidence_engine = TemporalConfidenceEngine(window_size=5, base_beta=0.5, decay_k=20.0, device=device) 
    blender = AdaptiveBlender(blur_k=15, device=device)
    metrics = ObjectiveMetricsEngine(device=device)
    
    # FAZ 5: Entropi Kodlayıcısı Başlatılıyor
    entropy_codec = TriBandEntropyCodec(device=device)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width * 2, orig_height))

    frame_idx = 0
    clean_plate_gpu = torch.zeros((3, orig_height, orig_width), device=device, dtype=torch.float32)
    clean_plate_gpu[0, :, :], clean_plate_gpu[1, :, :], clean_plate_gpu[2, :, :] = 25/255.0, 25/255.0, 30/255.0 

    gray_weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=torch.float16 if 'cuda' in device else torch.float32).view(3, 1, 1)
    kx = torch.tensor([[[-1., 0., 1.]]], device=device).view(1,1,1,3)
    ky = torch.tensor([[[-1.], [0.], [1.]]], device=device).view(1,1,3,1)
    gauss_kernel = torch.tensor([[[[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]]], device=device, dtype=torch.float32)
    
    use_pin = 'cuda' in device
    cpu_export_buffer = torch.empty((orig_height, orig_width, 3), dtype=torch.uint8, pin_memory=use_pin)
    
    prev_gray_gpu = None
    prev_edge_gpu = None
    prev_var_gpu = None
    
    ema_temp = torch.tensor(0.01, device=device, dtype=torch.float32)
    ema_edge = torch.tensor(0.01, device=device, dtype=torch.float32)
    ema_var = torch.tensor(0.01, device=device, dtype=torch.float32)
    
    EMA_ALPHA = 0.1
    HYBRID_SCENE_CUT_THRESHOLD = 3.5 
    
    drift_accumulator = torch.tensor(0.0, device=device, dtype=torch.float32)
    DRIFT_THRESHOLD = 10.0 

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_resized = cv2.resize(frame, (orig_width, orig_height))
        pts = frame_idx * (1.0 / fps)
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device).float().div(255.0)
        if 'cuda' in device: frame_t = frame_t.half()
        
        frame_t_squeeze = frame_t.squeeze(0)
        frame_gray_gpu = torch.sum(frame_t_squeeze * gray_weights, dim=0, keepdim=True)

        edge_x = F.conv2d(frame_gray_gpu.float().unsqueeze(0), kx, padding=(0,1))
        edge_y = F.conv2d(frame_gray_gpu.float().unsqueeze(0), ky, padding=(1,0))
        
        curr_edge = torch.sqrt(edge_x * edge_x + edge_y * edge_y + 1e-4)
        curr_edge = F.conv2d(curr_edge, gauss_kernel, padding=1).squeeze(0) 

        gray_unsqueezed = frame_gray_gpu.unsqueeze(0)
        gray_sq = gray_unsqueezed * gray_unsqueezed
        
        mean_val = F.avg_pool2d(gray_unsqueezed, kernel_size=16, stride=16)
        mean_sq = F.avg_pool2d(gray_sq, kernel_size=16, stride=16)
        curr_var = (mean_sq - (mean_val * mean_val)).squeeze(0).float()
        
        is_scene_cut = False
        hybrid_score = torch.tensor(0.0, device=device)
        
        if prev_gray_gpu is not None:
            raw_temp_diff = torch.mean(torch.abs(frame_gray_gpu - prev_gray_gpu))
            raw_edge_diff = torch.mean(torch.abs(curr_edge - prev_edge_gpu))
            raw_var_diff = torch.mean(torch.abs(curr_var - prev_var_gpu)) 
            
            ema_temp.mul_(1 - EMA_ALPHA).add_(raw_temp_diff * EMA_ALPHA)
            ema_edge.mul_(1 - EMA_ALPHA).add_(raw_edge_diff * EMA_ALPHA)
            ema_var.mul_(1 - EMA_ALPHA).add_(raw_var_diff * EMA_ALPHA)
            
            norm_temp = raw_temp_diff / (ema_temp + 1e-6)
            norm_edge = raw_edge_diff / (ema_edge + 1e-6)
            norm_var = raw_var_diff / (ema_var + 1e-6)
            
            hybrid_score = (0.4 * norm_var) + (0.3 * norm_temp) + (0.3 * norm_edge)
            
            is_cut_tensor = hybrid_score > HYBRID_SCENE_CUT_THRESHOLD
            if is_cut_tensor.item(): 
                is_scene_cut = True
                
            drift_accumulator.add_(hybrid_score * 0.1)
            
        if prev_gray_gpu is None:
            prev_gray_gpu = torch.empty_like(frame_gray_gpu)
            prev_edge_gpu = torch.empty_like(curr_edge)
            prev_var_gpu = torch.empty_like(curr_var)
            
        prev_gray_gpu.copy_(frame_gray_gpu)
        prev_edge_gpu.copy_(curr_edge)
        prev_var_gpu.copy_(curr_var)

        is_drift_refresh = False
        is_drift_tensor = drift_accumulator > DRIFT_THRESHOLD
        if is_drift_tensor.item() and frame_idx > 0:
            is_drift_refresh = True
            drift_accumulator.zero_()

        if is_scene_cut: 
            drift_accumulator.zero_()

        reset_rvm = (frame_idx == 0) or is_scene_cut or is_drift_refresh
        
        with torch.no_grad(): 
            rgb_gpu, pha_gpu = rvm.process_frame_tensor(frame_t, reset_state=reset_rvm)

        buffer.push(frame_idx, pts, rgb_gpu, pha_gpu)
        causal_window = buffer.get_causal_window()
        confidence_engine.compute_confidence(causal_window)

        if len(causal_window) > 0:
            current_state = causal_window[-1]
            dsre_final_tensor = blender.blend(current_state, clean_plate_gpu)
            
            alpha_mask = current_state.tensors.alpha_core
            conf_mask = current_state.tensors.confidence_map
            metrics.accumulate(frame_t_squeeze, dsre_final_tensor, alpha_mask, frame_gray_gpu)
            
            # FAZ 5: ENTROPI KODLAMASI (GERÇEK VERİ SIKIŞTIRMA)
            # Sadece arka planı silinmiş nesneyi (Residual RGB) ağa gönderiyoruz. (Siyah pikseller 0 entropidir)
            residual_rgb = frame_t_squeeze * alpha_mask
            
            # Tensörleri olasılık modelleyicisine yolla ve bayt boyutunu al
            frame_payload_kb = entropy_codec.encode_frame(residual_rgb, alpha_mask, conf_mask)
            
            out_tensor_half = (dsre_final_tensor * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            cpu_export_buffer.copy_(out_tensor_half, non_blocking=True)
            
            if use_pin:
                torch.cuda.current_stream().synchronize()
                
            out_numpy = cpu_export_buffer.numpy()
            dsre_final_render = cv2.cvtColor(out_numpy, cv2.COLOR_RGB2BGR)
            
            combined_frame = np.hstack((frame_resized, dsre_final_render))
            cv2.putText(combined_frame, "ORIJINAL (STANDART STREAM)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(combined_frame, f"DSRE FAZ 5 ({frame_payload_kb:.1f} KB/Frame)", (orig_width + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out.write(combined_frame)
            print(f"Frame {frame_idx} İşleniyor... (Entropy Coding: {frame_payload_kb:.1f} KB)", end="\r")

        frame_idx += 1

    cap.release()
    out.release()
    
    # --- FAZ 5 BİLİMSEL SONUÇ RAPORU ---
    metric_report = metrics.get_report()
    entropy_report = entropy_codec.get_report()
    
    print("\n\n" + "="*60)
    print(" 🚀 DSRE FAZ 5: NİHAİ KODEK (ENTROPY) RAPORU")
    print("="*60)
    print(f" İşlenen Toplam Kare Sayısı  : {metric_report['frames_evaluated']}")
    print(f" Ortalama BPP (Bits/Pixel)   : {entropy_report['avg_bpp']:.4f} bpp")
    print(f" Toplam Ağ Yükü (Payload)    : {entropy_report['total_payload_mb']:.2f} MB")
    print(f" Pseudo-Trimap PSNR          : {metric_report['avg_psnr']:.2f} dB")
    print(f" Hareket Telafili Instability: {metric_report['avg_mc_energy']:.6f}")
    print("-" * 60)
    print(" Not: Teorik Shannon Entropisi ve Olasılık Dağılımı (PDF) hesaplandı.")
    print(" Residual RGB (Sadece Ön Plan), Alpha ve Low-Res Confidence verileri")
    print(" CABAC/rANS sınırlarına göre bit seviyesinde simüle edilmiştir.")
    print(" Proje 'Semantik Filtre' seviyesinden 'Taşıma Katmanı' (Codec) seviyesine geçmiştir.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_real_video_test("test.mp4", "dsre_sonuc_final.mp4")
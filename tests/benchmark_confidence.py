import time
import numpy as np
import sys
import os

# 1. Python'a projenin ana dizinini (D:\Proje\DSRE) öğretiyoruz ki 'src' klasörünü bulabilsin.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Yazdığımız motorları içeri aktarıyoruz (İşte senin aldığın hatayı çözen kısım burası)
from src.core.temporal_dependency_buffer import CausalRingBuffer
from src.core.temporal_confidence import TemporalConfidenceEngine

def run_benchmark():
    print("\n[DSRE Benchmark] Architecture & Confidence Motoru Testi Başlatılıyor...\n")
    
    # Sistemleri Ayağa Kaldır
    buffer = CausalRingBuffer(width=1080, height=1920, capacity=16, active_window=5)
    confidence_engine = TemporalConfidenceEngine(base_beta=0.3)
    
    frames_count = 50
    start_time = time.perf_counter()
    
    print("-> Akış (Streaming) simülasyonu başlatıldı (50 Kare)...")
    
    for i in range(frames_count):
        # Statik kısımlar (Vücut = 1.0)
        alpha_dummy = np.zeros((1, 1920, 1080), dtype=np.float16)
        alpha_dummy[:, 500:1500, 300:700] = 1.0 
        
        # Dalgalanan/titreyen kısımlar (Saç uçları = rastgele noise)
        noise = np.random.uniform(0.2, 0.8, (1, 100, 100))
        alpha_dummy[:, 400:500, 300:400] = noise.astype(np.float16)
        
        # Boş RGB
        rgb_dummy = np.zeros((3, 1920, 1080), dtype=np.float16)
        
        # Pipeline 1: Kareyi Buffer'a al
        buffer.push(frame_idx=i, pts=i*0.033, rgb_data=rgb_dummy, alpha_data=alpha_dummy)
        
        # Pipeline 2: Buffer'dan pencereyi çek ve güveni hesapla
        causal_window = buffer.get_causal_window()
        confidence_engine.compute_confidence(causal_window)
        
        if i > 0 and i % 10 == 0:
            current_state = causal_window[-1]
            conf_map = current_state.tensors.confidence_map
            
            mean_conf = np.mean(conf_map)
            hair_conf = np.mean(conf_map[:, 400:500, 300:400])
            
            print(f"  [Frame {i:02d}] Genel Güven: %{mean_conf*100:.1f} | Problemli Saç Güveni: %{hair_conf*100:.1f}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    print("\n" + "="*50)
    print(f"[SONUÇ] Toplam 50 Kare İşlendi.")
    print(f"[SONUÇ] Toplam Süre: {elapsed_ms:.2f} ms")
    print(f"[SONUÇ] Frame Başına İşlem Süresi: {(elapsed_ms/frames_count):.2f} ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmark()
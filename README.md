# DSRE: State-Aware GPU-Resident Semantic Video Transport Pipeline

**DSRE (Dynamic Semantic Reconstruction Engine) Core V1.0** is an experimental, production-grade research platform focused on next-generation semantic video compression and transport pipelines for short-form and interactive video systems such as telepresence, livestreaming, gaming, and mobile-first media platforms.

Unlike conventional DCT-based codecs (H.264/H.265/AV1) that continuously re-encode static spatial information, DSRE separates semantic foreground data from redundant background regions, estimates the theoretical entropy of the meaningful signal, and reconstructs frames through a GPU-resident tensor pipeline optimized for low synchronization overhead and high temporal coherence.

---

# 🚀 Key Architectural Features

### • Tri-Band Entropy Modeling

Performs real-time Shannon Entropy estimation on sparse semantic residuals only:

* RGB residual stream
* Alpha matte
* Low-resolution confidence map

This avoids repeatedly encoding static scene regions and enables semantic payload estimation at significantly lower theoretical bandwidth.

---

### • Low-Synchronization CUDA Pipeline

The entire temporal analysis stack operates directly on VRAM-resident tensors:

* EMA smoothing
* Scene-cut detection
* Temporal variance analysis
* Confidence propagation

CPU ↔ GPU synchronization bottlenecks (`.item()` stalls) are intentionally minimized to preserve CUDA pipeline continuity.

---

### • Motion-Compensated Temporal Metrics

Implements FFT Phase Correlation to align sequential frames before difference analysis.

This isolates:

* genuine reconstruction instability,
* AI flicker,
* temporal artifacts,

from natural camera movement such as pans and handheld motion.

---

### • Boundary-Aware Reconstruction Metrics

Uses morphological dilation and trimap-based PSNR evaluation to measure quality specifically around:

* hair strands,
* motion blur regions,
* semi-transparent edges,
* fine transition boundaries.

---

### • Edge-Preserving Adaptive Blending

Replaces lossy average pooling approaches with an optimized separable Gaussian pipeline for isotropic edge-aware compositing and stable gradient preservation.

---

# 🧠 Pipeline Overview

## 1. Semantic Matting

Integrates Robust Video Matting (RVM) inference using FP16 precision for efficient foreground extraction.

## 2. Temporal Ring Buffer

Maintains a zero-allocation temporal history using:

* preallocated tensor pools,
* causal frame buffering,
* VRAM-conscious memory reuse.

## 3. Temporal Confidence Engine

Calculates spatial and temporal variance fully on the GPU to generate dynamic confidence maps at reduced resolution.

## 4. Entropy Codec (Phase 5)

Quantizes tensor data into 8-bit space and computes:

* probability density functions (PDF),
* Shannon entropy,
* estimated physical payload size.

## 5. Adaptive Reconstruction

Performs edge-aware alpha compositing for final frame reconstruction.

---

# 📂 Project Structure

```text
DSRE/
├── src/
│   ├── core/
│   │   ├── metrics_engine.py
│   │   ├── temporal_confidence.py
│   │   └── temporal_dependency_buffer.py
│   │
│   ├── external/
│   │   └── rvm_wrapper.py
│   │
│   ├── reconstruction/
│   │   └── adaptive_blender.py
│   │
│   ├── schemas/
│   │   └── frame_metadata.py
│   │
│   └── transport/
│       └── entropy_codec.py
│
└── main_runner.py
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/yourusername/DSRE.git
cd DSRE
```

## Install dependencies

> A CUDA-capable GPU and compatible PyTorch build are recommended.

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Place a video file named `test.mp4` inside the project root directory and run:

```bash
python main_runner.py
```

> `test.mp4` is intentionally excluded from the repository due to GitHub file size limitations.
> You may use any short video clip for testing, preferably containing a moving human subject.

---

# 📦 requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pydantic>=2.0.0
```

---

# 🔬 Research Status

DSRE Core V1.0 currently represents **Phase 5** of the broader DSRE research initiative.

The project functions as:

* a semantic transport architecture prototype,
* a probabilistic payload modeler,
* and a GPU pipeline optimization platform.

The current implementation successfully models the physical and informational bounds of semantic transport systems, but does not yet implement:

* CABAC,
* rANS,
* arithmetic entropy bitstream generation,
* hardware decoder interoperability.

---

# 🧩 Engineering Focus

The architecture is heavily optimized around:

* PyTorch Tensor Fusion
* VRAM allocation discipline
* CUDA Graph compatibility
* low-allocation temporal systems
* asynchronous export pipelines

---

# 📜 License

This repository is currently released for research and educational purposes.

Future revisions may introduce:

* experimental transport protocols,
* real-time streaming integrations,
* semantic caching systems,
* mobile-oriented inference backends.

---

# 🏆 Project Vision

DSRE explores the possibility of treating video not as a sequence of full images, but as a continuously evolving semantic state representation.

The long-term objective is to investigate whether semantic transport pipelines can substantially reduce bandwidth requirements while preserving perceptual continuity in modern interactive media systems.

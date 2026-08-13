# Unified Latent-State Memory Fabric (UL-SMF)
### Linear-Complexity KV Cache Compression via GLRP v2.0 & Aegis-KV

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

The **Unified Latent-State Memory Fabric (UL-SMF)** is a hardware-software co-designed memory compression fabric that solves the memory bottleneck in long-context Transformer inference. By combining **Finite Scalar Quantization (FSQ)** with dynamic 16-dimensional latent mapping, UL-SMF compresses Key-Value (KV) cache tensors by up to **384x** while maintaining **>94% semantic retention**.

---

## ⚠️ Enterprise & Commercial Licensing Notice

**UL-SMF is dual-licensed:**

1. **Open Source (AGPLv3):** Free for non-commercial research, academic use, and open-source projects. *Note: The AGPLv3 license requires any network-accessible service using this software to open-source its entire backend application code.*
2. **Commercial Enterprise License:** Required for proprietary commercial deployments, closed-source SaaS platforms, and enterprise data center infrastructure. Commercial licenses grant full rights without AGPLv3 copyleft restrictions, plus integration support.

📩 **For Enterprise Licensing Inquiries:** `inquiries@lawrencearchitectures.com`

---

## Key Benchmarks (Measured on CUDA Hardware)

| Metric | Raw FP32 Cache | UL-SMF 16D Latent | Improvement |
| :--- | :--- | :--- | :--- |
| **VRAM Footprint (4096 tokens)** | 48.00 MB | 0.12 MB | **384x Reduction** |
| **VRAM Saved / Block** | — | **47.88 MB** | **99.7% Memory Saved** |
| **Semantic Retention** | 100% | **94.15% - 95.84%** | Cosine Similarity |
| **Pipeline Latency** | — | **~14.1 ms - 19.6 ms** | CUDA Event Verified |

---

## Quickstart (2-Line Integration)

UL-SMF monkey-patches directly into standard PyTorch attention modules (Hugging Face Transformers, vLLM, etc.):

```python
import torch
from ul_smf import GLRPVersion2Bridge, UL_SMF_Interceptor

# 1. Initialize the Latent Bridge
bridge = GLRPVersion2Bridge(input_dim=3072, latent_dim=16).to("cuda")

# 2. Intercept and Compress the Attention Layer
model.model.layers[0].self_attn = UL_SMF_Interceptor(model.model.layers[0].self_attn, bridge)

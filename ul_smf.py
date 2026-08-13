import torch
import torch.nn as nn
import time

# --- 1. The Core Quantization & Bridge Architecture ---

class Version2FSQ(nn.Module):
    """Finite Scalar Quantization (FSQ) implementing the Version 2.0 4x4 discrete latent grid."""
    def __init__(self, levels: list[int] = [8, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3]):
        super().__init__()
        self.levels = levels
        self.register_buffer("half_levels", torch.tensor([(l - 1) / 2.0 for l in levels]))
        self.register_buffer("scale", torch.tensor([2.0 / (l - 1) for l in levels]))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = torch.tanh(z)
        half_l = self.half_levels.to(z.device)
        quantized_indices = torch.round(scaled * half_l + half_l)
        quantized_continuous = (quantized_indices - half_l) * self.scale.to(z.device)
        # Straight-Through Estimator (STE) for gradient preservation
        z_q = z + (quantized_continuous - z).detach()
        return z_q, quantized_indices


class GLRPVersion2Bridge(nn.Module):
    """GLRP v2.0 Asymmetric Autoencoder Bridge: Transduces 3072-float input tensors to 16 quantized integers."""
    def __init__(self, input_dim: int = 3072, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim)
        )
        self.fsq = Version2FSQ()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, indices = self.fsq(z)
        reconstructed = self.decoder(z_q)
        return reconstructed, indices


# --- 2. The Interceptor Wrapper ---

class UL_SMF_Interceptor(nn.Module):
    """
    Unified Latent-State Memory Fabric (UL-SMF) PyTorch Wrapper.
    Intercepts standard Attention KV caches and routes them through 
    the GLRP v2.0 / Aegis-KV finite scalar quantization bridge.
    """
    def __init__(self, original_attention, ul_smf_bridge):
        super().__init__()
        self.original_attn = original_attention
        self.ul_smf = ul_smf_bridge
        
        if hasattr(original_attention, 'config'):
            self.config = original_attention.config

    def forward(self, hidden_states, *args, **kwargs):
        attn_outputs = self.original_attn(hidden_states, *args, **kwargs)
        
        if len(attn_outputs) > 1 and attn_outputs[1] is not None:
            k_cache, v_cache = attn_outputs[1]
            
            original_shape = k_cache.shape
            flat_k = k_cache.reshape(-1, 3072)
            
            _, latent_indices = self.ul_smf(flat_k)
            
            compressed_kv = (latent_indices, v_cache)
            attn_outputs = (attn_outputs[0], compressed_kv) + attn_outputs[2:]
            
        return attn_outputs

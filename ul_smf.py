import torch
import torch.nn as nn
from typing import Tuple, Optional

class UniversalLatentBridge(nn.Module):
    def __init__(self, core_module: nn.Module, core_dim: int = 128):
        super().__init__()
        self.core_module = core_module
        self.core_dim = core_dim
        self.in_proj: Optional[nn.Module] = None
        self.out_proj: Optional[nn.Module] = None
        self.active_dim: Optional[int] = None

    def _setup_projections(self, in_dim: int, device: torch.device, dtype: torch.dtype):
        if in_dim == self.core_dim:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            in_layer = nn.Linear(in_dim, self.core_dim, bias=False).to(device=device, dtype=torch.float32)
            out_layer = nn.Linear(self.core_dim, in_dim, bias=False).to(device=device, dtype=torch.float32)
            nn.init.orthogonal_(in_layer.weight)
            with torch.no_grad():
                out_layer.weight.copy_(in_layer.weight.T)
            self.in_proj = in_layer.to(dtype=dtype)
            self.out_proj = out_layer.to(dtype=dtype)
        self.active_dim = in_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        in_dim = original_shape[-1]
        x_flat = x.view(-1, in_dim)
        
        if self.active_dim != in_dim:
            self._setup_projections(in_dim, x.device, x.dtype)
            
        x_core = self.in_proj(x_flat) if not isinstance(self.in_proj, nn.Identity) else x_flat
        reconstructed_core, latents = self.core_module(x_core)
        reconstructed = self.out_proj(reconstructed_core) if not isinstance(self.out_proj, nn.Identity) else reconstructed_core
        
        return reconstructed.view(original_shape), latents


class UL_SMF_Interceptor(nn.Module):
    """
    Model-Agnostic UL-SMF Attention Interceptor.
    Automatically detects the model's native attention head dimension 
    to prevent cross-token semantic bleed across any architecture.
    """
    def __init__(self, original_attention: nn.Module, bridge: UniversalLatentBridge):
        super().__init__()
        self.original_attn = original_attention
        self.bridge = bridge
        self.head_dim = 128  # Default fallback

        if hasattr(original_attention, 'config'):
            self.config = original_attention.config
            detected_dim = getattr(self.config, 'head_dim', None)
            if detected_dim is None and hasattr(self.config, 'hidden_size') and hasattr(self.config, 'num_attention_heads'):
                detected_dim = self.config.hidden_size // self.config.num_attention_heads
            if detected_dim is not None:
                self.head_dim = detected_dim

    def forward(self, hidden_states, *args, **kwargs):
        attn_outputs = self.original_attn(hidden_states, *args, **kwargs)

        if len(attn_outputs) > 1 and attn_outputs[1] is not None:
            k_cache, v_cache = attn_outputs[1]
            original_shape = k_cache.shape
            original_dtype = k_cache.dtype
            
            flat_heads = k_cache.reshape(-1, self.head_dim).to(torch.float32)
            reconstructed_heads, _ = self.bridge(flat_heads)
            
            compressed_keys = reconstructed_heads.to(original_dtype).reshape(original_shape)
            compressed_kv = (compressed_keys, v_cache)
            attn_outputs = (attn_outputs[0], compressed_kv) + attn_outputs[2:]
            
        return attn_outputs

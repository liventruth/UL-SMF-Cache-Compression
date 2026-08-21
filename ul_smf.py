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
        self._compression_verified = False  # One-time console audit flag

        if hasattr(original_attention, 'config'):
            self.config = original_attention.config
            detected_dim = getattr(self.config, 'head_dim', None)
            if detected_dim is None and hasattr(self.config, 'hidden_size') and hasattr(self.config, 'num_attention_heads'):
                detected_dim = self.config.hidden_size // self.config.num_attention_heads
            if detected_dim is not None:
                self.head_dim = detected_dim

    def forward(self, hidden_states, *args, **kwargs):
        # 1. Execute original attention
        attn_outputs = self.original_attn(hidden_states, *args, **kwargs)

        # 2. Extract past_key_value (Index 2 in Hugging Face attention outputs)
        if len(attn_outputs) > 2 and attn_outputs[2] is not None:
            past_kv = attn_outputs[2]
            
            k_cache, v_cache = None, None
            
            # Ensure it is a standard tuple cache, not a modern HF DynamicCache object
            if isinstance(past_kv, tuple):
                k_cache, v_cache = past_kv
                
            if k_cache is not None and v_cache is not None:
                # Print physical proof to the console the first time this fires
                if not self._compression_verified:
                    print("\n[UL-SMF AUDIT] Interceptor active: Aegis-KV compressing BOTH Keys and Values.")
                    self._compression_verified = True

                original_shape_k = k_cache.shape
                original_shape_v = v_cache.shape
                original_dtype = k_cache.dtype
                
                # 3. Compress and Reconstruct Keys (K)
                flat_keys = k_cache.reshape(-1, self.head_dim).to(torch.float32)
                reconstructed_keys, _ = self.bridge(flat_keys)
                compressed_keys = reconstructed_keys.to(original_dtype).reshape(original_shape_k)
                
                # 4. Compress and Reconstruct Values (V)
                flat_values = v_cache.reshape(-1, self.head_dim).to(torch.float32)
                reconstructed_values, _ = self.bridge(flat_values)
                compressed_values = reconstructed_values.to(original_dtype).reshape(original_shape_v)
                
                # 5. Rebuild the KV tuple and output tuple
                compressed_kv = (compressed_keys, compressed_values)
                attn_outputs = (attn_outputs[0], attn_outputs[1], compressed_kv) + attn_outputs[3:]
            else:
                # Fail-safe warning if a non-tuple Cache object bypassed the system
                if not self._compression_verified:
                    print("\n[UL-SMF AUDIT] WARNING: Non-tuple Cache detected. Compression bypassed.")
                    self._compression_verified = True
                
        return attn_outputs
        

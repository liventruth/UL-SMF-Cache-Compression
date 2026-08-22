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
    def __init__(self, original_attention: nn.Module, bridge: UniversalLatentBridge):
        super().__init__()
        self.original_attn = original_attention
        self.bridge = bridge
        self.head_dim = 128  
        self._audit_fired = False  

        if hasattr(original_attention, 'config'):
            self.config = original_attention.config
            detected_dim = getattr(self.config, 'head_dim', None)
            if detected_dim is None and hasattr(self.config, 'hidden_size') and hasattr(self.config, 'num_attention_heads'):
                detected_dim = self.config.hidden_size // self.config.num_attention_heads
            if detected_dim is not None:
                self.head_dim = detected_dim

    def forward(self, *args, **kwargs):
        # 1. Intercept the DynamicCache object passed via kwargs
        cache = kwargs.get('past_key_values', None)
        
        if cache is not None and hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            layer_idx = getattr(self.original_attn, 'layer_idx', len(cache.key_cache) - 1)
            
            if 0 <= layer_idx < len(cache.key_cache):
                k_tensor = cache.key_cache[layer_idx]
                v_tensor = cache.value_cache[layer_idx]
                
                if k_tensor is not None and v_tensor is not None:
                    if not self._audit_fired:
                        print(f"\n[UL-SMF AUDIT] SUCCESS: DynamicCache intercepted at layer {layer_idx}. Compressing Key/Value tensors.")
                        self._audit_fired = True
                    
                    # Compress Keys
                    orig_shape_k = k_tensor.shape
                    orig_dtype = k_tensor.dtype
                    flat_k = k_tensor.reshape(-1, self.head_dim).to(torch.float32)
                    rec_k, _ = self.bridge(flat_k)
                    cache.key_cache[layer_idx] = rec_k.to(orig_dtype).reshape(orig_shape_k)
                    
                    # Compress Values
                    orig_shape_v = v_tensor.shape
                    flat_v = v_tensor.reshape(-1, self.head_dim).to(torch.float32)
                    rec_v, _ = self.bridge(flat_v)
                    cache.value_cache[layer_idx] = rec_v.to(orig_dtype).reshape(orig_shape_v)

        # 2. Proceed with normal attention execution
        return self.original_attn(*args, **kwargs)

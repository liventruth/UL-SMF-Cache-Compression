import torch
import torch.nn as nn
from typing import Tuple, Optional

class UniversalLatentBridge(nn.Module):
    def __init__(self, core_module: nn.Module, core_dim: int = 3072):
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
    def __init__(self, original_module: nn.Module, bridge: UniversalLatentBridge):
        super().__init__()
        self.original_module = original_module
        self.bridge = bridge

    def forward(self, *args, **kwargs):
        output = self.original_module(*args, **kwargs)
        return output

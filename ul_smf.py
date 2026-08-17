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
            self.in_proj = nn.Linear(in_dim, self.core_dim, bias=False).to(device=device, dtype=dtype)
            self.out_proj = nn.Linear(self.core_dim, in_dim, bias=False).to(device=device, dtype=dtype)
            nn.init.orthogonal_(self.in_proj.weight)
            with torch.no_grad():
                self.out_proj.weight.copy_(self.in_proj.weight.T)
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

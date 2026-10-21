import torch
import torch.nn as nn

def _init_raw(n_bands: int, init: str, rng: torch.Generator | None) -> nn.Parameter:
    """Initializes (L, L, L) parameters as `NormalizedMSFA` does, intended to replace it.
    RNG ends in the same state.
    """
    raw = nn.Parameter(torch.empty(n_bands, n_bands, n_bands))
    match init:
        case "uniform":
            nn.init.uniform_(raw, 0, 1, generator=rng)
        case "constant":
            nn.init.ones_(raw)
        case _:
            raise ValueError(f"unknown init {init!r}, expected 'uniform' or 'constant'")
    return raw

def _entropy(p: torch.Tensor) -> torch.Tensor:
    """Mean per-pixel entropy over the band axis. Clamped at the smallest normal float, so
    0 * log 0 gives 0 instead of NaN; identical to the unclamped form otherwise.
    """
    tiny = torch.finfo(p.dtype).tiny
    return -(p * torch.log(p.clamp_min(tiny))).sum(dim=0).mean()

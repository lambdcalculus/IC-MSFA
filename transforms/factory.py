import torch

from typing import Sequence

def make_gaussian_cmf(wls_in: Sequence[int], wls_out: Sequence[int], std: float = 3.0):
    """Generates a CMF emulating a filter array with Gaussian response."""
    wls_repeated = torch.tensor(wls_in).float().expand(len(wls_out), -1)
    cmf = torch.empty(len(wls_out), len(wls_in))

    for i, wl in enumerate(wls_out):
        cmf[i] = 1/(std * ((2 * torch.pi) ** 0.5)) * torch.exp( - (((wls_repeated[i] - wl)/10)**2 / (2 * (std**2))) )
    return cmf

def make_ideal_cmf(wls_in: Sequence[int], wls_out: Sequence[int]):
    """Generates a CMF emulating an ideal filter array."""
    wls_repeated = torch.tensor(wls_in).float().expand(len(wls_out), -1)
    cmf = torch.zeros(len(wls_out), len(wls_in)).float()

    for i, wl in enumerate(wls_out):
        idx = wls_repeated[i] == wl
        cmf[i, idx] = 1.0

    return cmf



import torch
import torch.nn as nn
import torch.nn.functional as F
from transforms.util import _ensure_batch

class WeightedBilinear(nn.Module):
    """
    Weighted Bilinear interpolation method. Given parameters `(n_1, n_2)`, uses the kernel

    (1/n_1 * [1, 2, ..., n_1-1, n_1, n_1-1, ..., 1].T) @ (1/n_2 * [1, 2, ..., n_2-1, n_2, n_2-1, ..., 1]) 

    for interpolation for all bands.

    If mosaicking was done with Brauer MSFA, `n_1` and `n_2` correspond to the height and width of the
    kernel used, respectively.

    Args:
        params (int or tuple[int]): determines the size of the kernel. `(n_1, n_2)` gives a kernel
        of shape `(2*n_1-1, 2*n_2-1)`. A single int `n` gives a kernel of size `2*n-1`.

    NOTE: this is only recommended for MSFAs where all bands have equal density, since it
    uses the same kernel for all bands.

    Based on: J. Brauers and T. Aach, "A color filter array based multispectral camera".
    """
    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__()

        if isinstance(size, tuple):
            n_1, n_2 = size
        else: # size is an int
            n_1, n_2 = size, size

        mat_1 = 1/n_1 * torch.cat([
            torch.linspace(1, n_1, steps=n_1),
            torch.linspace(n_1-1, 1, steps=n_1-1)
        ]).unsqueeze(0).T
        mat_2 = 1/n_2 * torch.cat([
            torch.linspace(1, n_2, steps=n_2),
            torch.linspace(n_2-1, 1, steps=n_2-1)
        ]).unsqueeze(0)

        kernel = (mat_1 @ mat_2).unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    @_ensure_batch
    def forward(self, img: torch.Tensor):
        n_bands = img.shape[1]
        kernels = self.kernel.expand(n_bands, -1, -1, -1)
        out = F.conv2d(img, kernels, padding='same', groups=n_bands)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel={self.kernel})"

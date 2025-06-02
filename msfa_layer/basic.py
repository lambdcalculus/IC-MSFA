import torch
import torch.nn as nn
import transforms.functional as F

from torch import Tensor
from transforms.util import _ensure_batch

class BasicMSFA(nn.Module):
    """
    Basic optimazable MSFA layer.
    """
    def __init__(self, n_bands: int):
        super().__init__()

        self.msfa = nn.Parameter(torch.empty(n_bands, n_bands, n_bands), requires_grad=True)
        nn.init.uniform_(self.msfa, 0, 1)

        # clamp after each gradient update
        constrain_hook = lambda _: self.msfa.clamp_(0, 1)
        self.msfa.register_post_accumulate_grad_hook(constrain_hook)

    @_ensure_batch
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (torch.Tensor): input tensor (number of channels must match MSFA)

        Returns:
            torch.Tensor: input tensor after filtering through MSFA (same shape as input)
        """
        img = F.apply_msfa(img, self.msfa)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_bands={self.msfa.shape[0]})"

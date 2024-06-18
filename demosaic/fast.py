import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class Fast(nn.Module):
    """
    Fast but naive demosaicking algorithm based on bilinear interpolation.
    Only works on square MSFAs.
    
    Args:
        msfa (torch.Tensor): MSFA used to demosaic (shape: C x N x N).
    """

    def __init__(self, msfa: Tensor) -> None:
        super(Fast, self).__init__()
        assert msfa.shape[1] == msfa.shape[2], "Received non-square MSFA."

        msfa_normalized = msfa.div(msfa.sum((1, 2), keepdim=True).expand(msfa.shape))
        self.kernel = nn.Parameter(msfa_normalized.unsqueeze(1), requires_grad=False)
        self.kernel_size = msfa.shape[1]

    def forward(self, raw: Tensor):
        """
        Args:
            raw (torch.Tensor): raw single-channel image(s) to demosaic (shape: B x 1 x W x H).

        Returns:
            torch.Tensor: demosaicked image (shape: B x C x W x H).
        """
        out = F.conv2d(raw, self.kernel, stride=self.kernel_size)
        out = F.interpolate(out, scale_factor=self.kernel_size, mode="bilinear", align_corners=False)
        return out

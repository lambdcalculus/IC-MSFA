import torch
import torch.nn as nn

from torch import Tensor # makes typing a little nicer

def split_n_features(x: Tensor, n: int):
    """Splits the tenosr's channels into n groups."""
    x_list = list(torch.chunk(x, n, dim=1))
    return x_list

class f_g_layer(nn.Module):
    """Single part of a RevBlock.

    Applies two Conv3d's with 3x3 kernel and padding 1, with leaky ReLU in the middle.
    The shape of the tensor is preserved throughout.

    Args:
        channels (int): number of input (and output) channels
    """

    def __init__(self, channels: int):
        super(f_g_layer, self).__init__()
        self.nn_layer = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.nn_layer(x)
        return x

class RevBlock(nn.Module):
    """A reversible block of the RevSCI.

    Splits the image into `n` features, and applies `f_g_layer` successively,
    concatenating the results as it goes. As such, there will be in total `in_channels // n` layers inside the block.
    The shape of the tensor is preserved throughout.

    Args:
        in_channels (int): number of input (and output) channels
        n (int): number of features to split into
    """

    def __init__(self, in_channels: int, n: int):
        super(RevBlock, self).__init__()
        self.f = nn.ModuleList()
        self.n = n
        self.ch = in_channels
        for _ in range(n):
            self.f.append(f_g_layer(in_channels // n))

    def forward(self, x: Tensor) -> Tensor:
        feats = split_n_features(x, self.n)
        h_new = feats[-1] + self.f[0](feats[0])
        h_curr = h_new
        for i in range(1, self.n):
            h_new = feats[-(i+1)] + self.f[i](h_new)
            h_curr = torch.cat([h_curr, h_new], dim=1)
        return h_curr

    def reverse(self, y: Tensor) -> Tensor:
        """Applies the reverse transformation from `forward`.

        Used for reverse mode training, generally with `torch.no_grad`.
        """
        l = split_n_features(y, self.n)
        h_new = l[-1] - self.f[-1](l[-2])
        h_curr = h_new
        for i in range(2, self.n):
            h_new = l[-i] - self.f[-i](l[-(i+1)])
            h_curr = torch.cat([h_curr, h_new], dim=1)

        # we slice h_curr to obtain the first layer
        h_new = l[0] - self.f[0](h_curr[:, 0:(self.ch // self.n), ::])
        h_curr = torch.cat([h_curr, h_new], dim=1)
        return h_curr

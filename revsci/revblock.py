import torch
import torch.nn as nn

def split_n_features(x: torch.Tensor, n: int):
    """
    Splits the channels into n groups.
    """
    x_list = list(torch.chunk(x, n, dim=1))
    return x_list

class f_g_layer(nn.Module):
    """
    Single part of the RevBlock.
    Applies two Conv3d's with leaky ReLU in the middle.
    """
    def __init__(self, ch: int):
        super(f_g_layer, self).__init__()
        self.nn_layer = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nn_layer(x)
        return x

class RevBlock(nn.Module):
    """
    A reversible block of the RevSCI.
    Splits the image into `n` features, and applies f_g_layer successively,
    concatenating the results as we go.
    
    TODO: document further
    """

    def __init__(self, in_ch: int, n: int):
        super(RevBlock, self).__init__()
        self.f = nn.ModuleList()
        self.n = n
        self.ch = in_ch
        for _ in range(n):
            self.f.append(f_g_layer(in_ch // n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = split_n_features(x, self.n)
        h_new = feats[-1] + self.f[0](feats[0])
        h_curr = h_new
        for i in range(1, self.n):
            h_new = feats[-(i+1)] + self.f[i](h_new)
            h_curr = torch.cat([h_curr, h_new], dim=1)
        return h_curr

    def reverse(self, y: torch.Tensor):
        l = split_n_features(y, self.n)
        h_new = l[-1] - self.f[-1](l[-2])
        h_curr = h_new
        for i in range(2, self.n):
            h_new = l[-i] - self.f[-i](l[-(i+1)])
            h_curr = torch.cat([h_curr, h_new], dim=1)

        # não entendi o uso de slicing aqui
        h_new = l[0] - self.f[0](h_curr[:, 0:(self.ch // self.n), ::])
        h_curr = torch.cat([h_curr, h_new], dim=1)
        return h_curr

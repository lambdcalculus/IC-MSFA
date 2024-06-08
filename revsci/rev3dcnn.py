import torch
import torch.nn as nn
from torch.optim import Optimizer
from .revblock import RevBlock
from transforms import Unflatten
from typing import Callable

class Rev3DCNN(nn.Module):
    """
    Implementation of RevSCI, from:
    Z. Cheng et al., "Memory-Efficient Network for Large-scale Video Compressive Sensing," doi: 10.1109/CVPR46437.2021.01598.
    And code based on: https://github.com/BoChenGroup/RevSCI-net.

    Args:
        msfa (torch.Tensor): MSFA to use against the raw data (shape: C x X x Y).
        n_blocks (int): number of reversible blocks.
        n_split (int): number of splits in each rev block.
    """
    def __init__(self, msfa: torch.Tensor, n_blocks: int, n_split: int):
        super(Rev3DCNN, self).__init__()

        # unflatten raw data
        self.unflatten = Unflatten(msfa)
        
        n_bands = msfa.shape[0]
        # encoding / feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_bands, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        
        # rev blocks
        self.layers = nn.ModuleList()
        for _ in range(n_blocks):
            self.layers.append(RevBlock(64, n_split))

        # decoding
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, n_bands, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, raw: torch.Tensor):
        """
        Args:
            raw (torch.Tensor): the raw single-channel image (shape: B x 1 x W x H).

        Returns:
            torch.Tensor: the demosaicked image (shape: B x C x W x H).
        """
        out = self.unflatten(raw)
        out = self.conv1(out.unsqueeze(2))
        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out).squeeze(2)
        return out

    def for_backward(self, raw: torch.Tensor, gt: torch.Tensor, loss: Callable, opt: Optimizer):
        """
        Reverse training. Uses less memory, but is slower.

        TODO: document more
        """
        out = self.unflatten(raw)

        with torch.no_grad():
            out1 = self.conv1(out)
            out2 = out1
            for layer in self.layers:
                out2 = layer(out2)
        out3 = out2.requires_grad_()
        out4 = self.conv2(out3)

        loss1 = loss(torch.squeeze(out4), gt)
        loss1.backward()
        current_state_grad = out3.grad

        out_current = out3
        for layer in reversed(self.layers):
            with torch.no_grad():
                out_pre = layer.reverse(out_current)
            out_pre.requires_grad_()
            out_cur = layer(out_pre)
            torch.autograd.backward(out_cur, grad_tensors=current_state_grad)
            current_state_grad = out_pre.grad
            out_current = out_pre

        out1 = self.conv1(out)
        out1.requires_grad_()
        torch.autograd.backward(out1, grad_tensors=current_state_grad)
        if opt != 0:
            opt.step()

        return out4, loss1

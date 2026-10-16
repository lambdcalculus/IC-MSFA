import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .revblock import RevBlock, rev_stack

from typing import Literal, Optional
from torch import Tensor # makes typing a little nicer

class Rev3DCNN(nn.Module):
    """Implementation of RevSCI, from:
    Z. Cheng et al., "Memory-Efficient Network for Large-scale Video Compressive Sensing," doi: 10.1109/CVPR46437.2021.01598.

    And code based on: https://github.com/BoChenGroup/RevSCI-net.

    Args:
        msfa (torch.Tensor): MSFA to use against the raw data (shape: C x X x Y).
        n_blocks (int): number of reversible blocks.
        n_split (int): number of splits in each rev block.
        mode (str): how gradients go through the rev blocks while training. "autograd" stores
            every activation; "reversible" stores only the last block's output and reconstructs
            the rest in the backward pass; "checkpoint" stores each block's input and recomputes
            the block (`torch.utils.checkpoint`). "checkpoint" is kept as a control: it
            recomputes like "reversible" but from the true block inputs, so comparing the two
            separates the reconstruction error from the recomputation itself. All three compute
            the same gradients, up to rounding.
        fp32_stream (bool | None): keep the rev blocks' inputs/outputs (the additive couplings)
            in fp32 under autocast, while the convolutions inside them still run in fp16. The
            reversible backward then reconstructs the block inputs to ~1e-10 instead of ~1e-3,
            and its gradients stay within ~1e-4 of autograd's instead of ~1e-3. False runs the
            whole stack in fp16 under AMP, as every run before this option existed did. None
            (the default) means True in "reversible" mode and False otherwise, so autograd and
            checkpoint runs keep the old numerics. No effect without autocast.
    """

    def __init__(self, n_blocks: int, n_split: int,
                 mode: Literal["autograd", "reversible", "checkpoint"] = "autograd",
                 fp32_stream: Optional[bool] = None):
        super().__init__()

        if mode not in ("autograd", "reversible", "checkpoint"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.fp32_stream = (mode == "reversible") if fp32_stream is None else fp32_stream

        # encoding / feature extraction
        self.conv1 = nn.Sequential(
            # input shape: B x 1 x D x W x H
            #         bands go here^
            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2),
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
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, mosaic: Tensor) -> Tensor:
        """
        Args:
            mosaic (Tensor): sparse mosaiced multispectral image (shape: B x C x W x H).

        Returns:
            Tensor: the demosaicked image (shape: B x C x W x H).
        """
        # input shape: B x 1 x D x W x H
        # bands go on D
        out = self.conv1(mosaic.unsqueeze(1))
        if self.fp32_stream and out.dtype in (torch.float16, torch.bfloat16):
            # the additions in RevBlock then promote to fp32; the convs still autocast to fp16
            out = out.float()

        if self.mode == "reversible" and torch.is_grad_enabled():
            out = rev_stack(out, self.layers)
        elif self.mode == "checkpoint" and torch.is_grad_enabled():
            for layer in self.layers:
                out = checkpoint(layer, out, use_reentrant=False)
        else:
            for layer in self.layers:
                out = layer(out)

        out = self.conv2(out).squeeze(1)
        return out

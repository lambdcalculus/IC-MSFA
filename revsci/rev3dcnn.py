import torch
import torch.nn as nn
from torch.optim import Optimizer
from .revblock import RevBlock
from transforms import Unflatten
from typing import Callable, Optional, Tuple

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
        out: torch.Tensor = self.unflatten(raw)
        out = self.conv1(out.unsqueeze(2))
        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out).squeeze(2)
        return out

    def for_backward(self,
                     raw: torch.Tensor,
                     gt: torch.Tensor,
                     loss_fn: Callable,
                     opt: Optional[Optimizer]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient training using reversability. Uses less memory, but takes longer.
        Executes one mini-batch backpropagation.

        Args:
            raw (torch.Tensor): batch to run (shape: B x 1 x W x H)
            gt (torch.Tensor): ground-truth/target (shape: B x C x W x H)
            loss_fn (Callable): loss function to use
            opt (torch.Optimizer): optimizer to use (optional)

        Returns:
            torch.Tensor: the model's predcition
            torch.Tensor: the loss from prediction vs. ground-truth
        """
        # TODO: it may be possible to optimize this further? as things are currently,
        # we're going through every layer except conv2 twice

        unf: torch.Tensor = self.unflatten(raw)

        # only compute grads for conv2
        with torch.no_grad():
            out: torch.Tensor = self.conv1(unf.unsqueeze(2))
            for layer in self.layers:
                out = layer(out)
        out = out.requires_grad_()
        pred: torch.Tensor = self.conv2(out)

        # back-propagate, only until right before conv2
        loss = loss_fn(pred.squeeze(2), gt)
        loss.backward()

        # setting up reversal
        out_curr = out
        last_grad = out.grad
        # we go through layers in reverse, saving only the gradients we need and
        # thus saving up on memory
        for layer in reversed(self.layers):
            # we reverse, so we can forward again and get the grads
            with torch.no_grad():
                out_pre = layer.reverse(out_curr)
            out_pre.requires_grad_()

            # the values on this tensor are the same as out_curr, but they have gradients now
            # TODO: maybe something more manual would be faster? we already have out_cur without
            # gradients, maybe there's a quicker way to set them up
            out_curr_with_grad: torch.Tensor = layer(out_pre)
            out_curr_with_grad.backward(gradient=last_grad)

            # set up next iteration
            last_grad = out_pre.grad
            out_curr = out_pre

        # now back-propagate conv1
        out = self.conv1(unf)
        out.requires_grad_()
        out.backward(gradient=last_grad)
        if opt:
            opt.step()

        return pred, loss

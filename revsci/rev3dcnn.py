import torch
import torch.nn as nn

from .revblock import RevBlock
from transforms import Unflatten

from typing import Optional
from collections.abc import Callable
from torch import Tensor # makes typing a little nicer
from torch.optim import Optimizer

class Rev3DCNN(nn.Module):
    """Implementation of RevSCI, from:
    Z. Cheng et al., "Memory-Efficient Network for Large-scale Video Compressive Sensing," doi: 10.1109/CVPR46437.2021.01598.

    And code based on: https://github.com/BoChenGroup/RevSCI-net.

    Args:
        msfa (torch.Tensor): MSFA to use against the raw data (shape: C x X x Y).
        n_blocks (int): number of reversible blocks.
        n_split (int): number of splits in each rev block.
    """

    def __init__(self, msfa: Tensor, n_blocks: int, n_split: int):
        super().__init__()

        # unflatten raw data
        self.unflatten = Unflatten(msfa)
        
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

    def forward(self, raw: Tensor) -> Tensor:
        """
        Args:
            raw (Tensor): the raw single-channel image (shape: B x 1 x W x H).

        Returns:
            Tensor: the demosaicked image (shape: B x C x W x H).
        """
        out: Tensor = self.unflatten(raw)

        # input shape: B x 1 x D x W x H
        # bands go on D
        out = self.conv1(out.unsqueeze(1))
        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out).squeeze(1)
        return out

    def for_backward(self,
                     raw: Tensor,
                     gt: Tensor,
                     loss_fn: Callable[[Tensor, Tensor], Tensor],
                     opt: Optional[Optimizer]) -> tuple[Tensor, Tensor]:
        """Memory-efficient training using reversability. Uses less memory, but takes longer.

        Executes backpropagation for one batch, stepping the optimizer, if passed.

        Args:
            raw (torch.Tensor): batch to run (shape: B x 1 x W x H)
            gt (torch.Tensor): ground-truth/target (shape: B x C x W x H)
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): loss function to use
            opt (torch.Optimizer): optimizer to use (optional)

        Returns:
            torch.Tensor: the model's predcition
            torch.Tensor: the loss from prediction vs. ground-truth
        """
        unf: Tensor = self.unflatten(raw).unsqueeze(1)

        # compute conv1, with grads. we keep it saved
        out_conv1: Tensor = self.conv1(unf)

        # skip grads for rev-blocks
        out = out_conv1
        with torch.no_grad():
            for layer in self.layers:
                out = layer(out)
        out = out.requires_grad_()

        # get grads for conv2, save the prediction
        pred: Tensor = self.conv2(out)

        # back-propagate, only until right before conv2
        loss = loss_fn(pred, gt.unsqueeze(1))
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
            out_curr_with_grad: Tensor = layer(out_pre)
            out_curr_with_grad.backward(gradient=last_grad)

            # set up next iteration
            last_grad = out_pre.grad
            out_curr = out_pre

        # then we do the back-prop for conv1
        out_conv1.backward(gradient=last_grad)

        # step optimizer
        if opt is not None:
            opt.step()
            opt.zero_grad()

        return pred.squeeze(1), loss

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
    `n` must divide `in_channels`. The shape of the tensor is preserved throughout

    TODO (?): make it so `n` doesn't need to divide in_channels

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

    def backward_pass(self, y: Tensor, grad_y: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Backward through the block from its output alone, without stored activations.

        RevNet's backward (Gomez et al., 2017, Alg. 1), extended to `n` groups: each `f` is
        evaluated once, with grad, on the same input it had in `forward`, and that evaluation
        also reconstructs the block's input. Cheaper than `reverse` followed by `forward`, which
        evaluates every `f` twice. Call it under the autocast state of the forward pass.

        Args:
            y (torch.Tensor): the block's output
            grad_y (torch.Tensor): gradient of the loss with respect to `y`

        Returns:
            torch.Tensor: the reconstructed input (detached)
            torch.Tensor: gradient of the loss with respect to the input
            list[torch.Tensor]: gradients for `self.parameters()`, in that order
        """
        n = self.n
        # forward was h[0] = x[n-1] + f[0](x[0]) and h[k] = x[n-1-k] + f[k](h[k-1]), k >= 1
        h = split_n_features(y.detach(), n)
        g = split_n_features(grad_y, n)
        x: list[Tensor] = [None] * n       # pyright: ignore[reportAssignmentType]
        grad_x: list[Tensor] = [None] * n  # pyright: ignore[reportAssignmentType]
        grad_f: list[tuple[Tensor, ...]] = [()] * n

        # gradient reaching h[k] in total; h[n-1] only leaves the block
        grad_h = g[n - 1]
        for k in range(n - 1, 0, -1):
            inp = h[k - 1].detach().requires_grad_()
            with torch.enable_grad():
                out = self.f[k](inp)
            x[n - 1 - k] = h[k] - out.detach()
            grad_x[n - 1 - k] = grad_h
            grads = torch.autograd.grad(out, (inp, *self.f[k].parameters()), grad_h.to(out.dtype))
            grad_f[k] = grads[1:]
            grad_h = g[k - 1] + grads[0]

        # x[0] was reconstructed in the loop (k = n-1); it also feeds f[0]
        inp = x[0].detach().requires_grad_()
        with torch.enable_grad():
            out = self.f[0](inp)
        x[n - 1] = h[0] - out.detach()
        grad_x[n - 1] = grad_h
        grads = torch.autograd.grad(out, (inp, *self.f[0].parameters()), grad_h.to(out.dtype))
        grad_f[0] = grads[1:]
        grad_x[0] = grad_x[0] + grads[0]

        param_grads = [gp for gf in grad_f for gp in gf]
        return torch.cat(x, dim=1), torch.cat(grad_x, dim=1), param_grads

class _RevStack(torch.autograd.Function):
    """Applies a stack of `RevBlock`s, keeping only the last output for the backward pass.

    The block parameters are passed in explicitly, so their gradients are returned through
    autograd (and not written into `.grad` from inside the backward). `loss.backward()`,
    `torch.autograd.grad` and `GradScaler` then work as with any other layer.

    The backward runs under the autocast state of the forward, with the autocast cache off:
    autocast caches the fp16 copy of every fp32 leaf that requires grad (meant for weights)
    until its context exits, and each reconstructed block input is such a leaf when the
    residual stream is fp32. With the cache on, all of them stay alive until the whole stack
    is done, which costs more memory than plain autograd.
    """

    @staticmethod
    def forward(ctx, x: Tensor, blocks: nn.ModuleList, *params: Tensor) -> Tensor:
        ctx.blocks = blocks
        ctx.autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        with torch.no_grad():
            for block in blocks:
                x = block(x)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        y, = ctx.saved_tensors
        enabled, dtype = ctx.autocast
        param_grads: list[list[Tensor]] = []
        with torch.autocast("cuda", enabled=enabled, dtype=dtype, cache_enabled=False):
            for block in reversed(ctx.blocks):
                y, grad_y, grads = block.backward_pass(y, grad_y)
                param_grads.append(grads)
        flat = [gp for grads in reversed(param_grads) for gp in grads]
        return (grad_y, None, *flat)

def rev_stack(x: Tensor, blocks: nn.ModuleList) -> Tensor:
    """Applies `blocks` in sequence with reversible (memory-efficient) backpropagation.

    Only the output of the last block is kept; every block's input is reconstructed from its
    output during the backward pass. Same result as `for b in blocks: x = b(x)`, up to the
    rounding error of that reconstruction.
    """
    if len(blocks) == 0:
        return x
    return _RevStack.apply(x, blocks, *blocks.parameters()) # pyright: ignore[reportReturnType]

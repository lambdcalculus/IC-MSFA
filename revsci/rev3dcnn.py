import torch
import torch.nn as nn
from .revblock import RevBlock
from ..transforms import Unflatten

class Rev3DCNN(nn.Module):
    """
    Implementation of RevSCI.

    TODO: document more.
    """
    def __init__(self, msfa: torch.Tensor, n_bands: int, n_blocks: int, n_split: int):
        """
        Rev3DCNN's constructor.
        
        :param torch.Tensor msfa: MSFA to use against the raw data
        :param int n_bands: Number of bands/channels
        :param int n_blocks: Number of RevBlocks
        :param int n_split: Number of features in each RevBlock  
        """
        super(Rev3DCNN, self).__init__()

        # unflatten raw data
        self.unflatten = Unflatten(msfa)
        
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
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, raw: torch.Tensor):
        out: torch.Tensor

        out = self.unflatten(raw)
        out = self.conv1(out)
        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out)
        return out

        # batch_size = meas_re.shape[0]
        # mask = self.mask.to(meas_re.device)
        # maskt = mask.expand([batch_size, args.B, args.size[0], args.size[1]])
        # maskt = maskt.mul(meas_re)
        # data = meas_re + maskt
        # out = self.conv1(torch.unsqueeze(data, 1))

        # for layer in self.layers:
        #     out = layer(out)

        # out = self.conv2(out)


    def for_backward(self, mask, meas_re, gt, loss, opt, args):
        batch_size = meas_re.shape[0]
        maskt = mask.expand([batch_size, args.B, args.size[0], args.size[1]])
        maskt = maskt.mul(meas_re)
        data = meas_re + maskt
        data = torch.unsqueeze(data, 1)

        with torch.no_grad():
            out1 = self.conv1(data)
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

        out1 = self.conv1(data)
        out1.requires_grad_()
        torch.autograd.backward(out1, grad_tensors=current_state_grad)
        if opt != 0:
            opt.step()

        return out4, loss1

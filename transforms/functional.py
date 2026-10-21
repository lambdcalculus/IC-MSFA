import torch
from torch import Tensor
from collections.abc import Sequence

def apply_illuminant(img: Tensor, illum: Tensor) -> Tensor:
    """Applies an illuminant.

    Applies a CIE illuminant <https://en.wikipedia.org/wiki/Standard_illuminant> to
    acquire luminance data from reflectance data.

    When flattened, `illum` should contain the illuminance for each channel in `img`,
    in order.

    Args:
        img: reflectance data (shape: B x C x W x H)
        illum: illuminant to use (C elements)

    Returns:
        luminance data (shape: B x C x W x H)
    """
    assert img.shape[1] == illum.numel(), f"illum has {illum.numel()} but image has {img.shape[1]} channels."

    illum_matrix = torch.diag(illum.flatten()) / illum.max() # normalized linear transform
    img_pixels = img.flatten(start_dim=2, end_dim=3)         # shape: B x C x num_pixels

    # matmul handles the batched input fine
    lum_pixels = illum_matrix.matmul(img_pixels) # shape: B x C x num_pixels
    return lum_pixels.reshape(img.shape)         # shape: B x C x W x H

def apply_cmf(img: Tensor, cmf: Tensor) -> Tensor:
    """Applies a color-matching function to an input luminance image.

    If the input luminance is of shape C x W x H, the CMF should emulate the
    response of an observer to C wave-lengths, having shape N x C for some N.
    For each pixel, the output is:
        ```
        pixel_out = cmf @ pixel_in
        ```
    This can be used to emulate ground-truth data for snapshot MSFA cameras, as
    long as the CMF matches the response of the filters used by the camera.
    
    In that case, the output is as follows: for each pixel, the i-th component is the
    reading that the sensor for that pixel would have if that sensor had the
    filter for i-th band over it.

    Args:
        img: luminance data (B x C x W x H)
        cmf: color-matching function (N x C)

    Returns:
        CMF response (B x N x W x H)
    """
    img_pixels = img.flatten(start_dim=2, end_dim=3) # shape: B x C x num_pixels

    # matmul handles the batched input fine
    gt_pixels = cmf.matmul(img_pixels) # shape: B x N x num_pixels
    return gt_pixels.reshape(img.shape[0], -1, img.shape[2], img.shape[3]) # shape: B x N x W x H

def select(img: Tensor, bands: Sequence[int]) -> Tensor:
    """Selects the specified channels from `img`.

    Args:
        img (torch.Tensor): input tensor (shape: B x C x W x H)
        bands (sequence of int): the bands to select

    Returns:
        Tensor: the image with only the selected bands (shape: B x len(bands) x W x H).
    """
    bands = list(bands)
    return img[:, bands, :, :]

def repeat_msfa(msfa: Tensor, size: Sequence[int]):
    """Repeats the MSFA to the specified size.

    Args:
        msfa: the MSFA to repeat (shape: C x X x Y)
        size: the desired size (W, H)

    Returns:
        repeated MSFA (shape: C x W x H).
    """
    assert len(size) == 2, f"`size` has {len(size)} elements, should have 2"
    # and expand to match the batches
    rx = size[0] // msfa.shape[1] + 1
    ry = size[1] // msfa.shape[2] + 1
    repeated = msfa.repeat(1, rx, ry)[:, :size[0], :size[1]]
    return repeated

def apply_msfa(img: Tensor, msfa: Tensor) -> Tensor:
    """Applies the passed MSFA to `img`.
    The number of channels of `img` and `msfa` must match.

    Args:
        img (torch.Tensor): input tensor (shape: B x C x W x H).
        msfa (torch.Tensor): MSFA to use (shape: C x W x H).

    Returns:
        torch.Tensor: the filtered image (shape: B x C x W x H).
    """
    assert img.shape[1] == msfa.shape[0], f"`img` has {img.shape[1]} channels while `msfa` has {msfa.shape[0]}."

    # We repeat the MSFA to the shape of the input.
    repeated = repeat_msfa(msfa, (img.shape[2], img.shape[3]))

    # and expand to match the batches
    repeated = repeated.unsqueeze(0).expand(img.shape[0], -1, -1, -1)

    return img * repeated

def apply_msfa_sum(img: Tensor, msfa: Tensor) -> Tensor:
    """Applies the passed MSFA the way a snapshot sensor measures (proposal eq. 2), then
    scatters the reading back into `C` channels.
    Each pixel reads one number, the sum over channels of `img * msfa`, and channel `c` of
    the output is that reading times `msfa[c]`. For a binary MSFA this is the same as
    `apply_msfa`. For a soft one the input still carries one number per pixel.

    Args:
        img (torch.Tensor): input tensor (shape: B x C x W x H).
        msfa (torch.Tensor): MSFA to use (shape: C x W x H).

    Returns:
        torch.Tensor: the scattered sensor reading (shape: B x C x W x H).
    """
    assert img.shape[1] == msfa.shape[0], f"`img` has {img.shape[1]} channels while `msfa` has {msfa.shape[0]}."

    repeated = repeat_msfa(msfa, (img.shape[2], img.shape[3])).unsqueeze(0)

    # the sensor reading (shape: B x 1 x W x H)
    reading = (img * repeated).sum(dim=1, keepdim=True)

    return reading * repeated

def flatten(img: Tensor) -> Tensor:
    """Flattens a tensor's channels. Intended for images that have been filtered
    through a MSFA.

    Args:
        img (torch.Tensor): input tensor (shape: B x C x W x H)

    Returns:
        torch.Tensor: the flattened image(s) (shape: B x 1 x W x H).
    """
    return img.sum(dim=1, keepdim=True)

def unflatten(img: Tensor, msfa: Tensor) -> Tensor:
    """Unflattens a single-channel tensor using the given MSFA.

    Args:
        img (torch.Tensor): input tensor (shape: B x 1 x W x H)
        msfa (torch.Tensor): MSFA to use (shape: C x W x H)

    Returns:
        torch.Tensor: the unflattened image(s) (shape: B x C x W x H).
    """
    assert img.shape[1] == 1, f"Expected `img` with 1 channel, got {img.shape[1]} channels."

    # We need to repeat the input to match the number of channels.
    repeated_x = img.expand(-1, msfa.shape[0], -1, -1)

    # We repeat the MSFA to the shape of the input.
    repeated_msfa = repeat_msfa(msfa, (img.shape[2], img.shape[3]))

    # and expand to match the batches
    repeated_msfa = repeated_msfa.unsqueeze(0).expand(img.shape[0], -1, -1, -1)

    return repeated_x * repeated_msfa

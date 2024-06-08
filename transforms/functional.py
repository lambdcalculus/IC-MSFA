from torch import Tensor
from typing import List

def select(img: Tensor, bands: List[int]) -> Tensor:
    """Selects the specified channels from `img`.

    Args:
        img (torch.Tensor): input tensor (shape: B x C x W x H)
        bands (List[int]): the bands to select

    Returns:
        Tensor: the image with only the selected bands (shape: B x C x W x H).
    """
    return img[:, bands, :, :]

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
    rx = img.shape[2] // msfa.shape[1] + 1
    ry = img.shape[3] // msfa.shape[2] + 1
    repeated = msfa.repeat(1, rx, ry)[:, :img.shape[2], :img.shape[3]]

    # and expand to match the batches
    repeated = repeated.unsqueeze(0).expand(img.shape[0], -1, -1, -1)

    return img * repeated

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
    rx = img.shape[2] // msfa.shape[1] + 1
    ry = img.shape[3] // msfa.shape[2] + 1
    repeated_msfa = msfa.repeat(1, rx, ry)[:, :img.shape[2], :img.shape[3]]

    # and expand to match the batches
    repeated_msfa = repeated_msfa.unsqueeze(0).expand(img.shape[0], -1, -1, -1)

    return repeated_x * repeated_msfa

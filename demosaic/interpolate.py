import torch
import numpy as np
from numpy.typing import NDArray

# TODO: look at scipy.interpolate.griddata, it has more options than just linear
from scipy.interpolate import LinearNDInterpolator

from .util import _ensure_numpy

class Interpolate:
    """
    Linearly interpolates each layer independently via SciPy's LinearNDInterpolator.

    Not implemented in Torch, and thus CPU-bound.

    TODO: look into using `scipy.interpolate.griddata` instead?
    """
    def __init__(self) -> None:
        pass

    @_ensure_numpy
    def __call__(self, img: NDArray) -> NDArray:
        n_bands, w, h = img.shape
        out = np.empty_like(img)

        # padding the image by 1 is necessary to avoid NaNs at the edges
        # we sample the maximum from a square (actually, half-square, because we're padding at the border) of side-length n_bands
        img = np.pad(img, ((0, 0), (1, 1), (1, 1)), 'maximum', stat_length=n_bands)
        for i in range(n_bands):
            img_i = img[i, :, :]
            non_zero = img_i > 0
            interp = LinearNDInterpolator(np.argwhere(non_zero), img_i[non_zero])

            img_i = img_i[1:-1, 1:-1] # remove the padding
            out[i] = interp(np.argwhere(img_i >= 0)).reshape(w, h)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


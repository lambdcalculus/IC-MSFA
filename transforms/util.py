from numpy import loadtxt # using this to avoid pandas as a dependency
from scipy.io import loadmat
from torch import from_numpy
from functools import wraps

from torch import Tensor
from typing import Any, TypeAlias
from collections.abc import Callable

def load_mat(filename: str) -> Tensor:
    """Loads .mat file as a torch.Tensor.

    This automatically deals with finding the actual object in the dict that
    scipy.io.loadmat returns, directly returning a torch.Tensor instead.
    If only metadata is found, returns an empty tensor.

    Args:
        filename: path to the .mat file (must include file extension)
    """
    mat_data: dict[str, Any] = loadmat(filename)

    out = Tensor()
    for k, v in mat_data.items():
        if not k.startswith("__"):
            out = from_numpy(v)
            break

    return out.float()

def load_csv(filename: str) -> Tensor:
    """Loads a .csv file as a (2D) torch Tensor.

    Args:
        filename: path to the .csv file (must include file extension)
    """
    return from_numpy(loadtxt(filename, delimiter=',')).float()

TransformMethod: TypeAlias = Callable[[Any, Tensor], Tensor]
def _ensure_batch(func: TransformMethod) -> TransformMethod:
    """Wraps functions that require batched inputs, allowing them to take non-batched inputs,
    and ensuring that the output is batched only when the input is batched.

    Meant to be used as a function decorator inside transform methods."""
    @wraps(func) # makes sure the docstrings and other metadata are preserved
    def batch_ensured(self, img: Tensor) -> Tensor:
        needs_squeeze = False
        if img.ndim <= 3:
            img = img.unsqueeze(dim=0)
            needs_squeeze = True

        img = func(self, img)

        if needs_squeeze:
            img = img.squeeze(dim=0)
        return img

    return batch_ensured

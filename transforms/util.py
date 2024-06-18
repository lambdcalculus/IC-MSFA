from torch import Tensor
from typing import Any
from collections.abc import Callable

def _ensure_batch(func: Callable[[Any, Tensor], Tensor]) -> Callable[[Any, Tensor], Tensor]:
    """Wraps functions that require batched inputs, allowing them to take non-batched inputs,
    and ensuring that the output is batched only when the input is batched.

    Meant to be used as a function decorator inside transform methods."""
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

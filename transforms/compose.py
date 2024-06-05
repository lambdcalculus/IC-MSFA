from torch import Tensor
from typing import List, Callable

class Compose():
    '''
    Composes a list of transforms into one.
    '''
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for transform in self.transforms:
            img = transform(img)
        return img

    def __str__(self) -> str:
        msg = "Composed transform of:"
        for i, transform in enumerate(self.transforms):
            msg += f"\n({i}): {str(transform)}"
        return msg

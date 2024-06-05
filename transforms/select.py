from torch import Tensor
from typing import List

class Select():
    '''
    Selects the band numbers specified at construction.
    '''
    def __init__(self, bands: List[int]) -> None:
        self.bands = bands

    def __call__(self, img: Tensor) -> Tensor:
        if len(img.size()) > 3:
            # we're dealing with a batch
            return img[:, self.bands, :, :]

        return img[self.bands, :, :]

    def __str__(self) -> str:
        return f"Select transform that picks bands: {self.bands}."

from torch import Tensor

class MSFA():
    '''
    Applies color filtering with the MSFA specified at construction.
    Can only be used on images that match the MSFA's number of bands.
    '''
    def __init__(self, msfa: Tensor) -> None:
        self.msfa = msfa

    def __call__(self, img: Tensor) -> Tensor:
        if len(img.size()) > 3:
            # we're dealing with a batch
            assert img.size()[1] == self.msfa.size()[0], f"Number of bands is {img.size()[0]}, should be {self.msfa.size()[0]}."
        else: 
            # it's a single image
            assert img.size()[0] == self.msfa.size()[0], f"Number of bands is {img.size()[0]}, should be {self.msfa.size()[0]}."

        # To apply, we repeat the MSFA to the size of the image.
        # Remembering that we can be taking in either [batch_size, B, W, H] or [B, W, H] for the image,
        # so we're using negative indices to access the width and height.
        rx = img.size()[-2] // self.msfa.size()[-2] + 1
        ry = img.size()[-1] // self.msfa.size()[-1] + 1
        repeated = self.msfa.repeat(1, rx, ry)[:, :img.size()[-2], :img.size()[-1]]

        # torch will deal with the broadcasting, if we have a batch
        return img * repeated

    def __str__(self) -> str:
        return f"MSFA with {self.msfa.size()[0]} bands."

class Flatten():
    '''
    Flattens a sparse multi-band image into a 2D image, resembling raw sensor data.
    This is intended to be used with images that have gone through an MSFA filter.
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, img: Tensor) -> Tensor:
        if len(img.size()) > 3:
            return img.sum(1)
        else:
            return img.sum(0)

    def __str__(self) -> str:
        return "Flattens multi-band images into 2D."

class Unflatten():
    '''
    Unflattens a 2D image into the appropriate channels based on a MSFA.
    This is intended to be used with images that are (or emulate) raw sensor data.

    TODO: support for batches
    '''
    def __init__(self, msfa: Tensor) -> None:
        self.msfa = msfa

    def __call__(self, img: Tensor) -> Tensor:
        # To apply, we repeat the MSFA to the size of the image.
        # Remembering that we can be taking in either [batch_size, W, H] or [W, H] for the image,
        # so we're using negative indices to access the width and height.
        rx = img.size()[-2] // self.msfa.size()[-2] + 1
        ry = img.size()[-1] // self.msfa.size()[-1] + 1
        repeated_msfa = self.msfa.repeat(1, rx, ry)[:, :img.size()[-2], :img.size()[-1]]

        # Then, we expand the image to match the number of bands.
        if len(img.size()) > 2:
            # dealing with a batch
            repeated_img = img.unsqueeze(1).expand(-1, self.msfa.size()[0], -1, -1)
        else:
            # single image
            repeated_img = img.expand(self.msfa.size()[0], -1, -1)

        return repeated_img * repeated_msfa

    def __str__(self) -> str:
        return f"Unflattens with a MSFA of {self.msfa.size()[0]} bands."

from torch import Tensor, from_numpy
from numpy import ndarray, array
from numpy.typing import NDArray

from typing import Callable, Any

def _ensure_numpy(func: Callable[[Any, NDArray], NDArray]) -> Callable[[Any, NDArray | Tensor], NDArray | Tensor]:
    """
    Wraps methods that take numpy arrays so that they can take torch tensors or numpy arrays,
    and return the same type as they receive.
    """
    def numpy_ensured(self, img: NDArray | Tensor) -> NDArray | Tensor:
        if isinstance(img, ndarray):
            return func(self, img)
        if not isinstance(img, Tensor):
            raise TypeError("input is neither numpy array or torch tensor")
        # if we're here, img is a torch.Tensor
        # make sure it is in batch format
        needs_squeeze = False
        if img.ndim <= 3:
            img = img.unsqueeze(dim=0)
            needs_squeeze = True
        
        # save the original device
        orig_device = img.device
        img_np: NDArray = img.cpu().numpy()

        # loop through the batch with a for loop. not fast, but works
        outs: list[NDArray] = []
        for i in range(img_np.shape[0]):
            outs.append(func(self, img_np[i]))
        img_np = array(outs)

        # bring back to torch
        img = from_numpy(img_np).to(orig_device)

        # reshape as required
        if needs_squeeze:
            img = img.squeeze(dim=0)
        return img

    return numpy_ensured

from matplotlib.pyplot import axes
import numpy as np

from collections.abc import Sequence
from typing import Any, Literal

class MSFA:
    """Wraps a Numpy array offering various utilities for handling MSFAs."""

    def __init__(self, msfa: np.ndarray, form: Literal["binary", "index", "index_from_1"] = "binary"):
        if form == "index" or form == "index_from_1":
            raise NotImplementedError("MSFA in indices form not implemented.")
        if form != "binary":
            raise ValueError("Invalid value for form. Must be 'binary', 'indices' or 'indices_from_1.")
        
        assert msfa.ndim == 3, "msfa must have three dimensions."

        self._msfa = msfa 

    def __getitem__(self, key):
        return self._msfa.__getitem__(key)

    def __setitem__(self, key, value):
        return self._msfa.__setitem__(key, value)

    @property
    def msfa(self, form: Literal["binary", "index", "index_from_1"] = "binary") -> np.ndarray:
        match form:
            case "index":
                return self.index_form
            case "index_from_1":
                return self.index_from_1_form
            case "binary":
                return self.binary_form
            case _:
                return self.binary_form

    @property
    def binary_form(self) -> np.ndarray:
        return self._msfa

    @property
    def index_form(self) -> np.ndarray:
        out = np.empty(self._msfa.shape[:2], dtype=int)
        
        for idx in np.ndindex(self._msfa.shape[:2]):
            out[idx] = np.argwhere(self._msfa[idx[0], idx[1]]).item()

        return out

    @property
    def index_from_1_form(self) -> np.ndarray:
        return self.index_form + 1

    def visualize(self, wls: Sequence[Any], cmap: str = "jet", min_wl: int = 420, max_wl: int = 720, fig=False):
        from matplotlib import pyplot as plt

        assert len(wls) == self._msfa.shape[2], f"Got {len(wls)}, MSFA has {self._msfa[2]} bands."

        out = np.zeros((self._msfa.shape[0], self._msfa.shape[1]), dtype=float)
        idx_form = self.index_form

        wls_np = (np.array(wls) - min_wl) / (max_wl - min_wl)

        for idx in np.ndindex(self._msfa.shape[:2]):
            out[idx] = wls_np[idx_form[idx].item()]

        if not fig:
            plt.imshow(out, cmap=cmap, vmin=0.0, vmax=1.0)
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.imshow(out, cmap=cmap, vmin=0.0, vmax=1.0)
            return fig

    def visualize_5(self, fig=False):
        from matplotlib import pyplot as plt
        colors = np.array([[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 165, 0], [255, 0, 0]]) 

        out = np.zeros((self._msfa.shape[0], self._msfa.shape[1], 3), dtype=int)
        idx_form = self.index_form

        for idx in np.ndindex(self._msfa.shape[:2]):
            out[idx] = colors[idx_form[idx].item()]

        if not fig:
            plt.imshow(out)
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.imshow(out)
            return fig

    def __repr__(self) -> str:
        return self._msfa.__repr__()

    def __str__(self) -> str:
        return self._msfa.__str__()



        


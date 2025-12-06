import numpy as np
from itertools import combinations

def gen_osp(n_bands: int) -> np.ndarray:
    """
    Generates MSFA based on the MSFA-OSP algorithm. Based on: (article).
    
    Args:
        n_bands (int): number of bands desired.

    Returns:
        numpy.ndarray: generated MSFA (bands in first dimension).
    """

    # using numbers co-prime with n_bands guarantees uniformity later on
    candidates = [i for i in range(1, n_bands) if np.gcd(i, n_bands) == 1]

    W = np.zeros((n_bands, n_bands))

    for i, j in combinations(candidates, 2):
        I = np.ones((n_bands, 1)) @ np.arange(1, n_bands+1).reshape((1, n_bands))
        J = I.T
        B = (i * I + j * J) % n_bands + 1

        p = [np.array([coords[0], coords[1], B[coords]])
             for coords in np.ndindex(n_bands, n_bands)]

        D = [((pair[0] - pair[1])**2).sum() for pair in combinations(p, 2)]

        W[i-1, j-1] = np.min(D)

    a, b = np.unravel_index(W.argmax(), W.shape)
    I = np.ones((n_bands, 1), dtype=int) @ np.arange(1, n_bands+1).reshape((1, n_bands))
    J = I.T
    B = ((a+1) * I + (b+1) * J) % n_bands + 1

    msfa = np.zeros((n_bands, n_bands, n_bands))
    for idx in np.ndindex(n_bands, n_bands):
        msfa[B[idx]-1][idx] = 1

    return msfa


        


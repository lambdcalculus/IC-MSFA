import numpy as np

def rand_msfa(n_bands: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    msfa = np.zeros((n_bands, n_bands, n_bands), dtype=float)

    for idx in np.ndindex(n_bands, n_bands):
        i = rng.integers(0, n_bands)
        msfa[i, idx[0], idx[1]] = 1.0

    return msfa
    


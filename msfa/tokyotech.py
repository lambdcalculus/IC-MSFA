from numpy import array
from .msfa_class import MSFA

# TokyoTech 5-band MSFA. Bands are in order from shortest to longest wave-length, i.e.: B, Cy, G, Or, R
# TODO: link article
RGBCO = MSFA(array([
    [[0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]],
    [[1, 0, 1, 0, 1],
     [0, 1, 0, 1, 0],
     [1, 0, 1, 0, 1],
     [0, 1, 0, 1, 0],
     [1, 0, 1, 0, 1]],
    [[0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0]],
    [[0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0]],
], dtype=float))

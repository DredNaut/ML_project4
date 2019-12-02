
import matplotlib.pyplot as plot
import numpy as np
import os
from pca import *
from compress import *


X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
print(X.shape)
Z = compute_Z(X, scaling = True)
print(Z.shape)
COV = compute_covariance_matrix(Z)
print(COV.shape)
L, PCS = find_pcs(COV)
print(L.shape, PCS.shape)
Z_star = project_data(Z, PCS, L, 1, 0)
print(Z_star)

DATA = load_data('Project4-2/Project4/Data/Train/')
#compress_images(DATA, 10)
compress_images(DATA, 100)
#compress_images(DATA, 500)
#compress_images(DATA, 1000)
#compress_images(DATA, 2000)

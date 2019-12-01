import pca
import numpy as np

# Test Matricies
#X = np.array([[-1, -1],[-1,1],[1,-1],[1,1]])
X = np.array([[5,-5],[3,-3],[4,-3],[1,-2]])
# Data from the video, centered.
X = np.array([[1,1],[1,0],[2,2],[2,1],[2,4],[3,4],[3,3],[3,2],[4,4],[4,5],[5,5],[5,7],[5,4]])


# Centering Test
#Z = pca.compute_Z(X,True,False)
#print("[RESULT] After centering Value found for Z = ", Z)
#COV = pca.compute_covariance_matrix(Z)
#print("[RESULT] COV = ", COV)
#L, PCS = pca.find_pcs(COV)
#print("[RESULT] L = ", L)
#print("[RESULT] PCS = ", PCS)
#Z_star = pca.project_data(Z, PCS, L, 1, 0)

# Scaling Test
#Z = pca.compute_Z(X,False,True)
#print("[RESULT] After scaling Value found for Z = ", Z)
#COV = pca.compute_covariance_matrix(Z)
#print("[RESULT] COV = ", COV)
#L, PCS = pca.find_pcs(COV)
#print("[RESULT] L = ", L)
#print("[RESULT] PCS = ", PCS)
#Z_star = pca.project_data(Z, PCS, L, 1, 0)

# Centering and Scaling Test
Z = pca.compute_Z(X,True,True)
#print("[RESULT] After scaling and centering Value found for Z = ", Z)
COV = pca.compute_covariance_matrix(Z)
#print("[RESULT] COV = ", COV)
L, PCS = pca.find_pcs(COV)
#print("[RESULT] L = ", L)
#print("[RESULT] PCS = ", PCS)
Z_star = pca.project_data(Z, PCS, L, 1, 0)

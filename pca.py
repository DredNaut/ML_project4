import numpy as np
from matplotlib import pyplot


# The above function will take the data matrix X, and boolean variables centering and scaling. X has one
# sample per row. Remember there are no labels in PCA. If centering is True, you will subtract the mean
# from each feature. If scaling is True, you will divide each feature by its standard deviation. This function
# returns the Z matrix (numpy array), which is the same size as X.
def compute_Z(X, centering=True, scaling=False):
#	print("[TEST] computing z")

	if centering:
#		print("[TEST] Starting centering, subtracting mean from each feature")
		X_avg = np.mean(X, axis=0)
#		print("[TEST] X_avg = ",X_avg)
		Z = X - X_avg
#		print("[TEST] After centering Z = ",Z)


	if scaling:
#		print("[TEST] Starting scaling")
		Z = X/np.std(X)
#		print("[TEST] After scaling Z = ",Z)

	if centering and scaling:
#		print("[TEST] Starting centering, subtracting mean from each feature")
		X_avg = np.mean(X, axis=0)
		print("[TEST] X_avg = ",X_avg)
		Z = X - X_avg
		Z = np.round(Z, decimals=2)
		print("[TEST] After centering Z = ",Z)
		print("[TEST] Starting scaling")
		Z = Z/np.std(Z,axis=0)
		Z = np.round(Z, decimals=2)
		print("[TEST] After scaling Z = ",Z)



	return Z

# The above function will take the standardized data matrix Z and return the covariance matrix Z T Z=COV (a
# numpy array).
def compute_covariance_matrix(Z):
#	print("[TEST] Computing covariance matrix")
	print("ZT = \n",np.transpose(Z))
	print("Z = \n",Z)
	return np.dot(np.transpose(Z),Z)


# The above function will take the covariance matrix COV and return the ordered (largest to smallest) principal
# components PCS (a numpy array where each column is an eigenvector) and corresponding eigenvalues L (a
# numpy array). You will want to use np.linalg.eig for this.
def find_pcs(COV):
#	print("[TEST] Find pcs")
	L, PCS = np.linalg.eig(COV)
#	print("EiganValues = ",L)
#	print("EiganVectors = ",PCS)

## POSIBLY NORMALIZE
	return L, PCS


# The above function will take the standardized data matrix Z, the principal components PCS, and correspond-
# ing eigenvalues L, as well as a k integer value and a var floating point value. k is the number of principal
# components you wish to maintain when projecting the data into the new space. 0 ≤ k ≤ D. If k= 0, then we
# use the cumulative variance to determine the projection dimension. var is the desired cumulative variance
# explained by the projection. 0 ≤v≤ 1. If v= 0, then k is used instead. Assume they are never both 0 or
# both > 0. This function will return Z_star, the projected data.
def project_data(Z, PCS, L, k, var):
	print("[TEST] Project data")
	print("Z",Z)
	proj1 = np.dot(Z,PCS[0])
	proj2 = np.dot(Z,PCS[1])
	print("proj1 = ",proj1)
	print("proj2 = ",proj2)

	return (proj1,proj2)

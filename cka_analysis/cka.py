import numpy as np
from numba import jit, prange
from scipy.spatial.distance import cdist

@jit(nopython=True, parallel=True)
def center_kernel(K):
    """Center the kernel matrix."""
    K = K.astype(np.float32)  # Ensure consistent dtype
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.ones((n, n), dtype=np.float32) / n
    # Perform matrix multiplications
    KH = np.dot(K, H)
    centered_K = np.dot(H, KH)
    return centered_K

@jit(nopython=True, parallel=True)
def cka(K1, K2):
    """Compute the Centered Kernel Alignment (CKA) between two kernel matrices."""
    K1 = K1.astype(np.float32)  # Ensure consistent dtype
    K2 = K2.astype(np.float32)  # Ensure consistent dtype

    K1_centered = center_kernel(K1)
    K2_centered = center_kernel(K2)

    numerator = np.sum(K1_centered * K2_centered)
    denominator = np.sqrt(np.sum(K1_centered ** 2) * np.sum(K2_centered ** 2))

    return numerator / denominator

def linear_kernel(X):
    """Compute the linear kernel matrix."""
    return X @ X.T

def rbf_kernel(X, gamma=None):
    """Compute the RBF (Gaussian) kernel matrix."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    # Use scipy for optimized pairwise squared distances
    sq_dists = cdist(X, X, metric='sqeuclidean')
    return np.exp(-gamma * sq_dists)

# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 50).astype(np.float32)  # Use consistent dtype
    Y = np.random.rand(100, 50).astype(np.float32)  # Use consistent dtype

    K1 = linear_kernel(X)
    K2 = linear_kernel(Y)

    cka_value = cka(K1, K2)
    print(f"CKA value: {cka_value}")

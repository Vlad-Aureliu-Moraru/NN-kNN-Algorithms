import numpy as np

def lanczos(A, K, max_iter=None):
    n = A.shape[0]
    if max_iter is None:
        max_iter = 2*K

    # operator C = A A^T
    def C_mult(v):
        return A @ (A.T @ v)

    V = np.zeros((n, max_iter+1))
    alpha = np.zeros(max_iter)
    beta = np.zeros(max_iter+1)

    # initial vector
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    V[:, 0] = v

    # Lanczos iterations
    m = 0
    for j in range(max_iter):
        w = C_mult(V[:, j])
        if j > 0:
            w -= beta[j] * V[:, j-1]

        alpha[j] = np.dot(V[:, j], w)
        w -= alpha[j] * V[:, j]

        beta[j+1] = np.linalg.norm(w)
        if beta[j+1] < 1e-12:
            m = j + 1
            break

        V[:, j+1] = w / beta[j+1]
        m = j + 1

    # build small tridiagonal matrix T (m x m)
    T = (np.diag(alpha[:m]) +
         np.diag(beta[1:m], 1) +
         np.diag(beta[1:m], -1))

    # eigen-decomposition of T
    eigvals_small, eigvecs_small = np.linalg.eigh(T)

    # take top-K
    idx = np.argsort(eigvals_small)[::-1][:K]

    eigvals = eigvals_small[idx]
    eigvecs = V[:, :m] @ eigvecs_small[:, idx]

    return eigvals, eigvecs

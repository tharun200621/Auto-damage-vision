import numpy as np

# Load data
X = np.loadtxt("spice_locations (1) (1).txt", delimiter=",")
def lisan_al_gaib(X, K, max_iters=100):
    # Random initialization of centers
    centers = X[np.random.choice(X.shape[0], K, replace=False)]
    labels = np.zeros(X.shape[0], dtype=int)

    for _ in range(max_iters):  
        old_labels = labels.copy()

        #  E-step 
        distances = np.linalg.norm(
            X[:, None, :] - centers[None, :, :],
            axis=2
        )
        labels = np.argmin(distances, axis=1)

        # M-step 
        one_hot = np.eye(K)[labels]
        centers = (one_hot.T @ X) / one_hot.sum(axis=0)[:, None]

        # Convergence check
        if np.array_equal(labels, old_labels):
            break

    return centers, labels
K = 2  
centers, labels = lisan_al_gaib(X, K)

print("Final centers:")
print(centers)

import numpy as np
import time

def eigenfaces(A, K):
    start = time.time()
    mean_face = np.mean(A, axis=1)   
    A_centered = A - mean_face[:, np.newaxis]   
    L = np.dot(A_centered.T, A_centered)
    eigvals, eigvecs_small = np.linalg.eig(L)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_small = eigvecs_small[:, idx]

    eigenfaces = np.dot(A_centered, eigvecs_small)  

    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])
    eigenfaces = eigenfaces[:, :K]
    projections = np.dot(A_centered.T, eigenfaces)
    end = time.time()

    return end-start,mean_face, eigenfaces, projections


def eigenfaces_class_representatives(A, labels_A, K):

    start = time.time()
    mean_face = np.mean(A, axis=1)            
    A_centered = A - mean_face[:, None]      
    C = A_centered.T @ A_centered
    eigenvalues, eigenvectors_small = np.linalg.eig(C)  
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors_small = eigenvectors_small[:, idx]
    eigenfaces_matrix = A_centered @ eigenvectors_small
    eigenfaces_matrix = eigenfaces_matrix[:, :K]
    for i in range(K):
        eigenfaces_matrix[:, i] /= np.linalg.norm(eigenfaces_matrix[:, i])
    training_projections = eigenfaces_matrix.T @ A_centered   

    class_representatives = {}

    unique_classes = np.unique(labels_A)

    for c in unique_classes:
        idx_list = [i for i, label in enumerate(labels_A) if label == c]

        class_rep = np.mean(training_projections[:, idx_list], axis=1)

        class_representatives[c] = class_rep   

    end = time.time()

    classes = sorted(class_representatives.keys())
    rep_matrix = np.column_stack([class_representatives[c] for c in classes])

    return end-start, mean_face, eigenfaces_matrix, training_projections, rep_matrix, classes

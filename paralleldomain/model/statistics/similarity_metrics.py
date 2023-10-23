from typing import Optional, Union

import numpy as np
from numpy.linalg import norm


def entropy(a, b):
    """compute entropy but return 0 if p is <= 0."""
    if a <= 0:
        return 0
    if b <= 0:
        return np.inf

    return a * np.log(a / b)


def KL_divergence(A: np.ndarray, B: np.ndarray, reduction: Optional[str] = "sum") -> Union[float, np.ndarray]:
    """
    Compute the divergence score (KL divergence) between two distributions
    (higher = more divergence).
    Args:
        A (np.ndarray): discrete probability distribution P
        B (np.ndarray): discrete probability distribution Q
        P and Q must each sum to 1.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'sum' | 'mean'.
            'none': the output is element-wise
            'sum': the output will be summed
            'mean': the output will be divided by the number of elements in the output
            Default: 'mean'

    Returns:
        (Union[float, np.ndarray]): dissimilarity score

    """
    P, Q = A, B

    if reduction not in ["none", "sum", "mean"]:
        raise ValueError(f"reduction '{reduction}' not supported!")

    if P.shape != Q.shape:
        raise ValueError(
            f"distribution P (len: {P.shape}) and distribution Q (len: {Q.shape}) should have the same size."
        )

    # check to make sure probability distributions sum to 1.
    if not (np.isclose(P.sum(), 1) and np.isclose(Q.sum(), 1)):
        raise ValueError(f"P and Q must each sum to 1, P_sum = {P.sum()}, Q_sum = {Q.sum()}")

    kl_div = np.array([entropy(p, q) if q > 0 else np.inf for p, q in zip(P, Q)])

    if np.inf in kl_div and reduction is not None:
        return np.inf

    if reduction == "mean":
        return np.mean(kl_div)
    elif reduction == "sum":
        return np.sum(kl_div)
    else:
        return kl_div


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute cosine similarity between two distribution a, b. This measure whether two vectors (in our case distribution)
    points in the same direction when the scale differs.
    (higher = more alike).
    Args:
        a (np.ndarray): some distribution (don't require to be normalized)
        b (np.ndarray): some distribution (don't require to be normalized)

    Returns:
        (float): similarity score

    """
    return (A @ B.T) / (norm(A) * norm(B))


# def wassertein_dist():
#     pass

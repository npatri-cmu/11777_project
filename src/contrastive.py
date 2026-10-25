import numpy as np

def combine(original, distorted, alpha):
    x, y = np.asarray(original, dtype=float), np.asarray(distorted, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not len(x):
        raise ValueError('Scores must be matching nonempty vectors')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha < 0:
        raise ValueError('Invalid scores/alpha')
    return x.copy() if alpha == 0 else (1+alpha)*x-alpha*y

def predict(candidates, scores):
    if len(candidates) != len(scores):
        raise ValueError('Candidate/score mismatch')
    return candidates[int(np.argmax(scores))]

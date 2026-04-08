import sys
import numpy as np

def distance_matrix(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)

    D = np.empty((len(p1), len(p2)))

    dsin = np.sin(0.5 * (p1[:, None] - p2)) ** 2
    cosprod = np.cos(p1[:, 0])[:, None] * np.cos(p2[:, 0])
    a = dsin[:, :, 0] + cosprod * dsin[:, :, 1]
    D = 2 * np.arcsin(np.sqrt(a))
    
    D *= 6371  # Earth radius in km
    return D


def load_points(fname):
    data = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(1, 2))
    return data


def distance_stats(D):
    # Extract upper triangular part to avoid duplicate entries
    assert D.shape[0] == D.shape[1], 'D must be square'
    idx = np.triu_indices(D.shape[0], k=1)
    distances = D[idx]
    return {
        'mean': float(distances.mean()),
        'std': float(distances.std()),
        'max': float(distances.max()),
        'min': float(distances.min()),
    }


fname = sys.argv[1]
points = load_points(fname)
D = distance_matrix(points, points)
stats = distance_stats(D)
print(stats)
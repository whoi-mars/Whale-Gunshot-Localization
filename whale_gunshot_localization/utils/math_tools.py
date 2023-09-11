import itertools

import numpy as np

def point_circle_shortest_distance(center, r, point):
    """
    Calculate the shortest distance between a circle and point.

    Parameters
    ----------
    center : array-like 
        center of circle in form [y, x]
    r : float
        radius of circle
    point : array-like
        center of circle in form [y, x]

    Returns
    -------
    float
        closest distance between the provided point and circle
    """

    return np.abs(np.sqrt(((center - point) ** 2).sum()) - r)

def matrix_similarity(A, B):
    """
    Calculates similarity metric for two matrices based on the average
    cosine similarity of all corresponding rows and columns of the matrices.

    Parameters
    ----------
    A : array-like of shape M X N with M, N > 1
        input matrix
    B : array-like of shape M X N with M, N > 1
        input matrix

    Returns
    -------
    float
        matrix similarity metric in [0, 1]
    """

    # make matrices positive
    A = np.abs(A)
    B = np.abs(B)
    
    # row cosine distances
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    row_sim = (An * Bn).sum(axis=1)
    
    # column cosine distances
    An = A / np.linalg.norm(A, axis=0, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=0, keepdims=True)
    col_sim = (An * Bn).sum(axis=0)

    return np.mean([row_sim, col_sim])

def check_intersection(center_list, r_list):
    """
    Given multiple circles, determine if there exists any pair among them
    which do not interesect or are not tangent.

    Parameters
    ----------
    center_list : array-like of shape N X 2
        matrix of N circle centers of format [y, x]
    r_list : array-like of shape N,
        list of radii ordered so that they are in the same
        position as the associated circle center

    Returns
    -------
    bool
        whether all pairs of circles intersect or are tangent (True) or not (False)
    """
    
    # get pairwise combinations of ranges and sensors
    r_combos = list(map(list, itertools.combinations(r_list, 2)))
    center_combos = list(map(list, itertools.combinations(center_list, 2)))

    for r_combo, center_combo in zip(r_combos, center_combos):

        # distance between circle centers
        d = np.sqrt(((center_combo[0] - center_combo[1]) ** 2).sum())

        if d <= r_combo[0] - r_combo[1]:
            # second circle in first
            return False
        elif d <= r_combo[1] - r_combo[0]:
            # first circle in second
            return False
        elif d <= r_combo[0] + r_combo[1]:
            # circles intersect or are tangent
            continue
        else:
            # circles do not intersect
            return False
    return True

def is_sparse_locs(locs, thresh=2000):
    """
    Checks if a group of locations is sufficiently sparse such that
    all pairs of locations are greater than `thresh` meters apart.

    Parameters
    ----------
    locs : array-like of shape N X 2
        matrix of 2D locations
    thresh : float
        sparseness threshold

    Returns
    -------
    : bool
        Whether the locations are sufficiently spares (True) or not (False)
    """
    
    if locs.shape[0] > 1:
        for (l1, l2) in itertools.combinations(locs, 2):
            if np.sqrt(((l1 - l2) ** 2).sum()) < thresh:
                return False
    return True

def is_dense_locs(locs, thresh=6000, n=3):
    """
    Checks if a group of locations is sufficiently dense such that
    to every sensor, there are n others that are less than `thresh` 
    meters apart.

    Parameters
    ----------
    locs : array-like of shape N X 2
        matrix of 2D locations
    thresh : float
        closeness threshold
    n : int
        number of locs each loc should be sufficiently close to

    Returns
    -------
    : bool
        Whether the locations are sufficiently dense (True) or not (False)
    """
    for loc in locs:
        distances = np.sqrt(((locs - loc) ** 2).sum(axis=1))
        if (distances <= thresh).sum() < min(n + 1, len(locs)):
            return False
    return True

"""
Module for skeleton pruning, used during the estimation of the medial axis
of the segments.
"""
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize


def skeleton_connectivity(skeleton):
    """Compute the connectivity map of the skeleton (8-neighbouring).

    Parameters
    ----------
    skeleton : numpy.ndarray
        Input skeleton.

    Returns
    -------
    numpy.ndarray
        Connectivity map.
    """
    connectivity = np.zeros(skeleton.shape)
    i, j = np.nonzero(skeleton > 0)

    for oi in range(-1, 2):
        for oj in range(-1, 2):
            connectivity[i, j] += skeleton[i + oi, j + oj]

    return connectivity


def prune_min(skeleton):
    """Prune the skeleton by keeping only the longest branch.

    The branches are successively removed by the shortest first.

    Parameters
    ----------
    skeleton : numpy.ndarray
        Input skeleton.

    Returns
    -------
    numpy.ndarray
        Pruned skeleton.
    """
    def _prune_step(skeleton, labeled):
        """One pruning step (removing the shortest branch).

        Parameters
        ----------
        skeleton : numpy.ndarray
            Input skeleton.

        labeled : numpy.ndarray
            Connectivity map.

        Returns
        -------
        numpy.ndarray
            Pruned skeleton.
        """
        l_min = 1
        size_min = (labeled == l_min).sum()

        for l in range(2, labeled.max() + 1):
            size = (labeled == l).sum()

            if size < size_min:
                size_min = size
                l_min = l

        return np.bitwise_xor(skeleton, labeled == l_min)

    labeled = label(np.bitwise_and(skeleton > 0,
                                   skeleton_connectivity(skeleton) < 4))

    while labeled.max() > 1:
        skeleton = skeletonize(_prune_step(skeleton, labeled))
        labeled = label(np.bitwise_and(skeleton > 0,
                                       skeleton_connectivity(skeleton) < 4))

    return skeleton

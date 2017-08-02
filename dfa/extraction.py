"""
Extraction module of the DNA fiber analysis package.

Use this module to compare fiber paths, extract fiber profiles and images.
"""
import numpy as np


def coarse_fibers_spatial_distance(f1, f2):
    """
    Coarse spatial distance between two fibers.

    The coarse distance is computed as the euclidian distance between the
    centers of mass of the considered fibers.

    :param f1: First fiber to compare.
    :type f1: numpy.ndarray

    :param f2: Second fiber to compare.
    :type f2: numpy.ndarray

    :return: The coarse spatial distance between fibers (in spatial units).
    :rtype: float
    """
    cm_f1 = f1.mean(axis=1)
    cm_f2 = f2.mean(axis=1)

    return np.linalg.norm(cm_f1 - cm_f2, ord=2)


def coarse_fibers_orientation_distance(f1, f2):
    """
    Coarse orientation distance between two fibers.

    The global orientations of fibers are computed and compared.

    :param f1: First fiber to compare.
    :type f1: numpy.ndarray

    :param f2: Second fiber to compare.
    :type f2: numpy.ndarray

    :return: The coarse orientation distance between fibers (in degrees).
    :rtype: float
    """
    orient_f1 = f1[:, -1] - f1[:, 0]
    orient_f2 = f2[:, -1] - f2[:, 0]

    angle = np.abs((orient_f1 * orient_f2).sum() /
                   (np.linalg.norm(orient_f1, ord=2) *
                    np.linalg.norm(orient_f2, ord=2)))

    if angle > 1:
        angle = 1

    return 180 * np.arccos(angle) / np.pi


def match_fibers_pairs(l1, l2, max_spatial_distance=50,
                       max_orientation_distance=30):
    """
    Match pairs of fibers from two given lists.

    The coarse distance distance between fibers are computed and the
    generated distance map is traversed by minimal distance first to generate
    the pairs, until no pair can be created.

    The fibers are associated once. This means that if one list is bigger than
    the other, there will be some fibers from the biggest that will have
    no match in the other list.

    Also, the maximal distance parameters allow to not associate fibers that
    are to far away from each other and do not share a similar orientation.

    :param l1: First list of fibers.
    :type l1: list of numpy.ndarray

    :param l2: Second list of fibers.
    :type l2: list of numpy.ndarray

    :param max_spatial_distance: Maximal spatial distance accepted to be
    associated (in spatial units, default is 50).
    :type max_spatial_distance: positive float

    :param max_orientation_distance: Maximal orientation distance accepted to
    be associated (in degrees, default is 30).
    :type max_orientation_distance: float within range [0, 180[

    :return: The matched pairs of fibers.
    :rtype: list of tuples of numpy.ndarray
    """
    # Build distance map
    spatial_dist = np.zeros((len(l1), len(l2)))
    orientation_dist = np.zeros((len(l1), len(l2)))

    for i, f1 in enumerate(l1):
        for j, f2 in enumerate(l2):
            spatial_dist[i, j] = coarse_fibers_spatial_distance(f1, f2)
            orientation_dist[i, j] = coarse_fibers_orientation_distance(f1, f2)

    # Find pairs
    for k in range(min(spatial_dist.shape)):
        i, j = np.unravel_index(spatial_dist.argmin(), spatial_dist.shape)

        if spatial_dist[i, j] <= max_spatial_distance and \
           orientation_dist[i, j] <= max_orientation_distance:
            yield i, j
            spatial_dist[i, :] = spatial_dist.max()
            spatial_dist[:, j] = spatial_dist.max()
        else:
            break


def fibers_spatial_distances(f1, f2):
    """
    Pointwise spatial distance between two fibers.

    The distance returned are the mean of minimal distances between fibers
    in a pointwise manner and the modified Hausdorff distance.

    To make distances symmetric, the maximal values of both ways are taken as
    the final results.

    :param f1: First fiber to compare.
    :type f1: numpy.ndarray

    :param f2: Second fiber to compare.
    :type f2: numpy.ndarray

    :return: The spatial distances between fibers (in spatial units) (mean and
    Hausdorff).
    :rtype: tuple of floats
    """
    def _closest_distances(f1, f2):
        closest_distances = []

        for p in f1.T:
            min_distance = np.linalg.norm(p - f2[:, 0], ord=2)

            for q in f2.T[1:]:
                distance = np.linalg.norm(p - q, ord=2)

                if distance < min_distance:
                    min_distance = distance

            closest_distances.append(min_distance)

        return closest_distances

    closest_distances_f1 = _closest_distances(f1, f2)
    closest_distances_f2 = _closest_distances(f2, f1)

    return (max(np.mean(closest_distances_f1), np.mean(closest_distances_f2)),
            max(np.median(closest_distances_f1),
                np.median(closest_distances_f2)),
            max(np.max(closest_distances_f1), np.max(closest_distances_f2)))
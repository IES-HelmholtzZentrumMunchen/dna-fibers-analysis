"""
Compare module of the DNA fibers analysis package.

Use this module to compare and quantify the quality of the pipeline.
"""
import numpy as np


def coarse_fibers_spatial_distance(f1, f2):
    """
    Coarse spatial distance between two fibers.

    The coarse distance is computed as the euclidean distance between the
    centers of mass of the considered fibers.

    Parameters
    ----------
    f1 : numpy.ndarray
        First fiber to compare.

    f2 : numpy.ndarray
        Second fiber to compare.

    Returns
    -------
    float
        The coarse spatial distance between fibers (in spatial units).
    """
    cm_f1 = f1.mean(axis=1)
    cm_f2 = f2.mean(axis=1)

    return np.linalg.norm(cm_f1 - cm_f2, ord=2)


def coarse_fibers_orientation_distance(f1, f2):
    """
    Coarse orientation distance between two fibers.

    The global orientations of fibers are computed and compared.

    Parameters
    ----------
    f1 : numpy.ndarray
        First fiber to compare.

    f2 : numpy.ndarray
        Second fiber to compare.

    Returns
    -------
    float
        The coarse orientation distance between fibers (in degrees).
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

    Parameters
    ----------
    l1 : List[numpy.ndarray]
        First list of fibers.

    l2 : List[numpy.ndarray]
        Second list of fibers.

    max_spatial_distance : 0 <= float
        Maximal spatial distance accepted to be associated (in spatial units,
        default is 50).

    max_orientation_distance : 0 <= float < 180
        Maximal orientation distance accepted to be associated (in degrees,
        default is 30).

    Returns
    -------
    List[(numpy.ndarray, numpy.ndarray)]
        The matched pairs of fibers.
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
    Point-wise spatial distance between two fibers.

    The distance returned are the mean of minimal distances between fibers
    in a point-wise manner and the modified Hausdorff distance.

    To make distances symmetric, the maximal values of both ways are taken as
    the final results.

    Parameters
    ----------
    f1 : numpy.ndarray
        First fiber to compare.

    f2 : numpy.ndarray
        Second fiber to compare.

    Returns
    -------
    (float, float)
        The spatial distances between fibers (in spatial units) (mean and
        Hausdorff).
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


def match_index_pairs(d1, d2):
    """
    Match pairs of index from two given data frames.

    Internally, pandas methods are used to match unique index in both data
    frames. Here this method is used as a convenience method to match fibers
    in both data frames.

    The percentage of match is computed as the ratio of the number of identical
    index in both data frames over the maximal number of fibers in one data
    frame.

    Parameters
    ----------
    d1 : pandas.core.frame.DataFrame
        First data frame.

    d2 : pandas.core.frame.DataFrame
        Second data frame.

    Returns
    -------
    (float, pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)
        Percentage of match and views of input data frames where index are
        pair-wisely matching (in the same order as input arguments).
    """
    index1 = d1.index.unique()
    index2 = d2.index.unique()
    index = index1.intersection(index2)

    return (index.size / max(index1.size, index2.size),
            d1.ix[index], d2.ix[index])


def match_column(d1, d2, column='pattern'):
    """
    Match column values from two given indexed data frames.

    Internally, pandas methods are used to match columns. The index are
    considered as columns to match in the process. It is used as convenience
    method to check if matching fibers have also matching patterns.

    Parameters
    ----------
    d1 : pandas.core.frame.DataFrame
        First data frame.

    d2 : pandas.core.frame.DataFrame
        Second data frame.

    column : str
        Label of the column that will be matched (default is 'pattern').

    Returns
    -------
    (float, pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)
        Percentage of match and views of input data frames where columns (and
        index) are pair-wisely matching (in the same order as input arguments).
    """
    single_column1 = d1.reset_index()[
        np.bitwise_not(
            d1.reset_index().duplicated(d1.index.names + [column]))] \
        .set_index(d1.index.names)[column]

    single_column2 = d2.reset_index()[
        np.bitwise_not(
            d2.reset_index().duplicated(d2.index.names + [column]))] \
        .set_index(d2.index.names)[column]

    select = np.equal(single_column1, single_column2)

    return np.sum(select) / single_column1.size, d1[select], d2[select]


def difference_in_column(d1, d2, column='length'):
    """
    Compute the differences between given data frames in given column values.

    This is convenience method that can be used to compute the errors in lengths
    between two data frames.

    Parameters
    ----------
    d1 : pandas.core.frame.DataFrame
        First data frame.

    d2 : pandas.core.frame.DataFrame
        Second data frame.

    column : str
        Label of the column from which difference is computed (default is
        'length').

    Returns
    -------
    pandas.core.series.Series
        Difference between the two given data frames (d1-d2).
    """
    return d1[column] - d2[column]

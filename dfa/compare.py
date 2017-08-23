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
    (float, float, float)
        The spatial distances between fibers (in spatial units) (mean, median
        and Hausdorff).
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


def match_index_pairs(d1, d2, matches, columns=('expected fiber',
                                                'actual fiber')):
    """
    Match pairs of index from two given data frames.

    The fibers in data frames are matched using a list of fibers match, that
    can be the index of the output of fiber matching method. Here this method
    is used as a convenience method to match fibers in both data frames.

    The percentage of match is computed as the ratio of the number of identical
    index in both data frames over the maximal number of fibers in one data
    frame.

    Parameters
    ----------
    d1 : pandas.core.frame.DataFrame
        First data frame.

    d2 : pandas.core.frame.DataFrame
        Second data frame.

    matches : pandas.indexes.multi.MultiIndex
        Index giving the matches between data frames. It must contain the image
        identifier scheme (by default 'experiment' and 'image') and columns
        'expected fiber' and 'actual fiber' giving the fiber id.

    columns : List[str] or (str, str)
        Names of the columns corresponding respectively to d1 and d2 in index
        fibers_match.

    Returns
    -------
    (float, pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)
        Percentage of match and views of input data frames where index are
        pair-wisely matching (in the same order as input arguments).
    """
    index1 = matches.droplevel(columns[1])
    index1.names = index1.names[:-1] + ['fiber']

    index2 = matches.droplevel(columns[0])
    index2.names = index2.names[:-1] + ['fiber']

    return matches.size / max(d1.index.unique().size,
                              d2.index.unique().size), \
        d1.ix[index1], d2.ix[index2]


def match_column(d1, d2, column='pattern'):
    """
    Match column values from two given indexed data frames.

    Data frames are assumed to have index that are pairwise matching.

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
    d1_full = d1.reset_index()
    single_column1 = d1_full[
        np.bitwise_not(d1_full.duplicated(d1.index.names + [column]))] \
        .set_index(d1.index.names)[column].reset_index()

    d2_full = d2.reset_index()
    single_column2 = d2_full[
        np.bitwise_not(d2_full.duplicated(d2.index.names + [column]))] \
        .set_index(d2.index.names)[column].reset_index()

    select = np.array(single_column1['pattern'].tolist()) == \
        np.array(single_column2['pattern'].tolist())

    index1 = single_column1.set_index(d1.index.names)[select].index
    index2 = single_column2.set_index(d2.index.names)[select].index

    return np.sum(select) / len(select), d1.ix[index1], d2.ix[index2]


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
    differences = d1.reset_index(drop=True)[column] - \
        d2.reset_index(drop=True)[column]

    differences.name = 'differences of ' + column

    return differences

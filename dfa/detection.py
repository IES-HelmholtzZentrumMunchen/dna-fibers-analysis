"""
Detection module of the DNA fiber analysis package.

Use this module to detect fibers.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.measure import label
from skimage.morphology import skeletonize, binary_dilation, disk, white_tophat

from dfa import _grayscale_morphology as _gm
from dfa import _scale_space_hessian as _sha
from dfa import _skeleton_pruning as _sk
from dfa import _structuring_segments as _ss


def fiberness_filter(image, scales, alpha=0.5, beta=1.0, gamma=1):
    """Enhance fiber image using a multi-scale vesselness filter.

    This vesselness is based on the Frangi filter [1]_.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to filter.

    scales : list of float or iterable of float
        Sizes range.

    alpha : float between 0 and 1
        Soft threshold of the tubular shape weighting term. Default is the
        recommended value (i.e. 0.5).

    beta : positive float
        Soft threshold of the intensity response (intensity dynamic-dependent)
        as a percentage of an automatically estimated parameter. Default is
        1.0, i.e. the parameter as it is automatically estimated.

    gamma : positive float
        Gamma-normalization of the derivatives (see scale-space theory). If no
        scale is preferred, it should be set to 1 (default).

    Returns
    -------
    tuple of numpy.ndarray
        The multiscale fiberness map and the corresponding vector field of the
        directions with the less intensity variation.

    Notes
    -----
    .. [1] Frangi et al. (1998) Multi-scale vessel enhancement filtering. In:
       *Medical Image Computing and Computer-Assisted Intervention*, vol. 1496,
       pp. 130-137.
    """
    fiberness = np.zeros((len(scales),) + image.shape)
    directions = np.zeros((len(scales), 2) + image.shape)

    for n, scale in enumerate(scales):
        hxx, hyy, hxy = _sha.single_scale_hessian(image, scale, gamma)
        (l1, l2), (v1, _) = _sha.hessian_eigen_decomposition(hxx, hyy, hxy)
        fiberness[n] = _sha.single_scale_vesselness(l1, l2, alpha, beta)
        directions[n] = v1

    ss, si, sj = fiberness.shape
    scale_selection = fiberness.argmax(axis=0)

    return (fiberness[scale_selection,
                      np.tile(np.arange(si).reshape(si, 1), (1, sj)),
                      np.tile(np.arange(sj).reshape(1, sj), (si, 1))],
            directions[scale_selection,
                       np.tile(np.arange(2).reshape(2, 1, 1), (1, si, sj)),
                       np.tile(np.arange(si).reshape(1, si, 1), (2, 1, sj)),
                       np.tile(np.arange(sj).reshape(1, 1, sj), (2, si, 1))])


def reconstruct_fibers(fiberness, directions, length, size, mask, extent_mask):
    """Reconstruct fibers disconnections from vesselness map and directions of
    less intensity variations.

    The reconstruction is based on morphological closing using variant
    structuring segment, which orientation is defined by the principal vector
    of the Hessian tensor [2]_.


    Parameters
    ----------
    fiberness : numpy.ndarray
        Input vesselness map.

    directions : numpy.ndarray
        Vector field of less intensity variations of input image.

    length : strictly positive int
        Length of the structuring segment.

    size : strictly positive int
        Thickness of the structuring segment.

    mask : numpy.ndarray
        Mask of where could be fibers. It is useful to speed up the process,
        for instance when not the whole image contains interesting fibers.

    extent_mask : numpy.ndarray
        Mask of where the reconstructed fibers could be. It should contain the
        fibers mask and the parts to be reconstructed.

    Returns
    -------
    numpy.ndarray
        The reconstructed vesselness map.

    Notes
    -----
    .. [2] Tankyevych et al. (2008) Curvilinear morpho-Hessian filter. In: *5th
       IEEE International Symposium on Biomedical Imaging: From Nano to Macro*,
       pp. 1011-1014.
    """
    # We need to flip the vectors components (x corresponds to j, and y to i)
    directions = np.flip(directions, axis=0)

    # Regularize the vector field with a simple morphological dilation
    segments = _ss.structuring_segments(directions, size, length, 0)
    regularized_directions = _gm.morphological_regularization(
        fiberness, directions, segments, mask)
    segments = _ss.structuring_segments(regularized_directions, size, length, 0)

    # Reconstruct by morphological closing
    return _gm.adjunct_varying_closing(fiberness, segments, mask, extent_mask)


def _order_skeleton_points(skeleton):
    """Gives the list of points corresponding to the skeleton, in the
    consecutive order.

    Parameters
    ----------
    skeleton : numpy.ndarray
        Input skeleton.

    Returns
    -------
    tuple of lists of int
        Lists of ordered coordinates values
    """
    connectivity = _sk.skeleton_connectivity(skeleton)
    endpoints = connectivity == 2
    j, i = np.meshgrid(range(skeleton.shape[1]), range(skeleton.shape[0]))
    x0, xn = j[endpoints]
    y0, yn = i[endpoints]
    skeleton[y0, x0] = 0
    x = [x0]
    y = [y0]
    oj, oi = np.meshgrid(range(-1, 2), range(-1, 2))

    while x[-1] != xn or y[-1] != yn:
        next_point = skeleton[y[-1] + oi, x[-1] + oj] > 0
        x.append(x[-1] + int(oj[next_point]))
        y.append(y[-1] + int(oi[next_point]))
        skeleton[y[-1], x[-1]] = 0

    return x, y


def estimate_medial_axis(reconstruction, threshold=0.5, smoothing=10,
                         min_length=30):
    """Estimate the medial axis of the detected fibers from the reconstructed
    fiberness map.

    This current implementation uses a parametric B-Spline fitting to estimate,
    separately, the medial axis of the fibers.

    Parameters
    ----------
    reconstruction : numpy.ndarray
        Reconstructed fiberness map.

    threshold : integer between 0 and 1
        Threshold to use with the fiberness map.

    smoothing : strictly positive int
        Smoothing of the B-Spline fitting (in pixels).

    min_length : strictly positive int
        Approximate minimal length of fibers in pixels (default is 30).

    Returns
    -------
    list of numpy.ndarray
        Coordinates of the medial axis lines of corresponding fibers.
    """
    # Threshold vesselness map and get connected components
    skeletons = skeletonize(reconstruction >= threshold)
    labels = label(skeletons)
    coordinates = []

    for l in range(1, labels.max() + 1):
        # fiber_skeleton = np.equal(labels, l)
        fiber_skeleton = _sk.prune_min(np.equal(labels, l))
        number_of_pixels = fiber_skeleton.sum()

        if number_of_pixels >= min_length:
            # we assume the skeleton has only one branch (it is pruned)
            # noinspection PyTupleAssignmentBalance
            splines, u = splprep(
                np.vstack(_order_skeleton_points(fiber_skeleton)), s=smoothing)
            u_sampled = np.linspace(u.min(), u.max(), number_of_pixels)
            x, y = splev(u_sampled, splines)
            coordinates.append(np.vstack((x, y)))

    return coordinates


def detect_fibers(image, scales, alpha, beta, length, size, smoothing,
                  min_length, fiberness_threshold=0.5, extent_mask=None):
    """Convenience method of the fibers detection pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    scales : list of float or iterable of float
        The fibers are searched within these scales.

    alpha : float between 0 and 1
        Soft threshold of the tubular shape weighting term. Default is the
        recommended value (i.e. 0.5).

    beta : positive float
        Soft threshold of the intensity response (intensity dynamic-dependent)
        as a percentage of an automatically estimated parameter.

    length : strictly positive int
        Length of the structuring segment.

    size : strictly positive int
        Thickness of the structuring segment.

    smoothing : strictly positive int
        Smoothing of the B-Spline fitting (in pixels).

    min_length : strictly positive int
        Approximate minimal length of fibers in pixels (default is 30).

    fiberness_threshold : float between 0 and 1
        Threshold used on the fiberness map (default is 0.5).

    extent_mask : numpy.ndarray or None
        Mask where the fibers will be detected.

    Returns
    -------
    list of numpy.ndarray
        Coordinates of the medial axis lines of corresponding fibers.
    """
    fiberness, directions = fiberness_filter(
        white_tophat(image, disk(max(scales))),
        scales=scales, alpha=alpha, beta=beta)

    if extent_mask is None:
        mask = fiberness >= fiberness_threshold
        extent_mask = binary_dilation(mask, disk(length))
    else:
        mask = extent_mask

    reconstructed_vesselness = reconstruct_fibers(
        fiberness, directions, length=length, size=size, mask=mask,
        extent_mask=extent_mask)

    coordinates = estimate_medial_axis(
        reconstructed_vesselness, smoothing=smoothing, min_length=min_length)

    return coordinates

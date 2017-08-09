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
    """
    Enhance fiber image using a multi-scale vesselness filter.

    This vesselness is based on the Frangi filter (Frangi et al. Multiscale
    vessel enhancement filtering (1998). In: Medical ImageComputing and
    Computer-Assisted Intervention, vol. 1496, pp. 130-137).

    :param image: Input image to filter.
    :type image: numpy.ndarray

    :param scales: Sizes range.
    :type scales: list of float or iterable of float

    :param alpha: Soft threshold of the tubular shape weighting term. Default is
    the recommended value (i.e. 0.5).
    :type alpha: float between 0 and 1

    :param beta: Soft threshold of the intensity response (intensity dynamic-
    dependent) as a percentage of an automatically estimated parameter. Default
    is 1.0, i.e. the parameter as it is automatically estimated.
    :type beta: positive float

    :param gamma: Gamma-normalization of the derivatives (see scale-space
    theory). If no scale is preferred, it should be set to 1 (default).
    :type gamma: positive float

    :return: The multiscale fiberness map and the corresponding vector field
    of the directions with the less intensity variation.
    :rtype: tuple of numpy.ndarray
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
    """
    Reconstruct fibers disconnections from vesselness map and directions of
    less intensity variations.

    The reconstruction is based on morphological closing using variant
    structuring segment. This algorithm is based on the following publication:
    Tankyevych et al. Curvilinear morpho-Hessian filter (2008). In: 5th IEEE
    International Symposium on Biomedical Imaging: From Nano to Macro, pp.
    1011-1014

    :param fiberness: Input vesselness map.
    :type fiberness: numpy.ndarray

    :param directions: Vector field of less intensity variations of input image.
    :type directions: numpy.ndarray

    :param length: Length of the structuring segment.
    :type length: strictly positive int

    :param size: Thickness of the structuring segment.
    :type size: strictly positive int

    :param mask:  Mask of where could be fibers. It is useful to speed up the
    process, for instance when not the whole image contains interesting fibers.
    :type mask: numpy.ndarray

    :param extent_mask: Mask of where the reconstructed fibers could be. It
    should contain the fibers mask and the parts to be reconstructed.
    :type extent_mask: numpy.ndarray

    :return: The reconstructed vesselness map.
    :rtype: numpy.ndarray
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
    """
    Gives the list of points corresponding to the skeleton, in the consecutive
    order.

    :param skeleton: Input skeleton.
    :type skeleton: numpy.ndarray

    :return: Lists of ordered coordinates values
    :rtype: tuple of lists of int
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
    """
    Estimate the medial axis of the detected fibers from the reconstructed
    fiberness map.

    This current implementation uses a parametric B-Spline fitting to estimate,
    separately, the medial axis of the fibers.

    :param reconstruction: Reconstructed fiberness map.
    :type reconstruction: numpy.ndarray

    :param threshold: Threshold to use with the fiberness map.
    :type threshold: integer between 0 and 1

    :param smoothing: Smoothing of the B-Spline fitting (in pixels).
    :type smoothing: strictly positive int

    :param min_length: Approximative minimal length of fibers in pixels
    (default is 30).
    :type min_length: strictly positive int

    :return: Coordinates of the medial axis lines of corresponding fibers.
    :rtype: list of numpy.ndarray
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
    """
    Convenience method of the fibers detection pipeline.

    :param image: Input image.
    :type image: numpy.ndarray

    :param scales: The fibers are searched within these scales.
    :type scales: list of float or iterable of float

    :param alpha: Soft threshold of the tubular shape weighting term. Default is
    the recommended value (i.e. 0.5).
    :type alpha: float between 0 and 1

    :param beta: Soft threshold of the intensity response (intensity dynamic-
    dependent) as a percentage of an automatically estimated parameter.
    :type beta: positive float

    :param length: Length of the structuring segment.
    :type length: strictly positive int

    :param size: Thickness of the structuring segment.
    :type size: strictly positive int

    :param smoothing: Smoothing of the B-Spline fitting (in pixels).
    :type smoothing: strictly positive int

    :param min_length: Approximative minimal length of fibers in pixels
    (default is 30).
    :type min_length: strictly positive int

    :param fiberness_threshold: Threshold used on the fiberness map (default
    is 0.5).
    :type fiberness_threshold: float between 0 and 1

    :param extent_mask: Mask where the fibers will be detected.
    :type extent_mask: numpy.ndarray or None

    :return: Coordinates of the medial axis lines of corresponding fibers.
    :rtype: list of numpy.ndarray
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

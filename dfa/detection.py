"""
Detection module of the DNA fiber analysis package.

Use this module to detect fibers and extract their profiles.
"""
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev

from dfa import _scale_space_hessian as _sha
from dfa import _structuring_segments as _ss
from dfa import _grayscale_morphology as _gm
from dfa import _skeleton_pruning as _sk


def fiberness_filter(image, scales, alpha=0.5, beta=0.5, gamma=1):
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

    :param beta: Soft threshold of the intensity response (intensity-dependent)
    as a percentage of maximal Hessian norm (as proposed in Frangi et al.).
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


def reconstruct_fibers(fiberness, directions, length, size):
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

    :return: The reconstructed vesselness map.
    :rtype: numpy.ndarray
    """
    # We need to flip the vectors components (x corresponds to j, and y to i)
    directions = np.flip(directions, axis=0)

    # Regularize the vector field with a simple morphological dilation
    segments = _ss.structuring_segments(directions, size, length, 0)
    regularized_directions = _gm.morphological_regularization(
        fiberness, directions, segments)
    segments = _ss.structuring_segments(regularized_directions, size, length, 0)

    # Reconstruct by morphological closing
    return _gm.adjunct_varying_closing(fiberness, segments)


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
    :rtype: list of tuples of float
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
            coordinates.append((x.reshape(x.size, 1),
                                y.reshape(y.size, 1)))

    return coordinates


def detect_fibers(image, scales, alpha, beta, length, size, smoothing, min_length):
    fiberness, directions = fiberness_filter(
        image, scales=scales, alpha=alpha, beta=beta)

    reconstructed_vesselness = reconstruct_fibers(
        fiberness, directions, length=length, size=size)

    coordinates = estimate_medial_axis(
        reconstructed_vesselness, smoothing=smoothing, min_length=min_length)

    return coordinates


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
    cm_f1 = f1.mean(axis=0)
    cm_f2 = f2.mean(axis=0)

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
    orient_f1 = f1[-1, :] - f1[0, :]
    orient_f2 = f2[-1, :] - f2[0, :]

    angle = (orient_f1 * orient_f2).sum() / \
            (np.linalg.norm(orient_f1, ord=2) *
             np.linalg.norm(orient_f2, ord=2))

    if angle > 1:
        angle = 1

    return 180 * np.arccos(angle) / np.pi


def match_fibers_pairs(l1, l2, max_spatial_distance=5,
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
    associated (in spatial units, default is 5).
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
            spatial_dist[i, j] = coarse_fibers_spatial_distance(
                np.vstack(f1).T, np.vstack(f2).T)
            orientation_dist[i, j] = coarse_fibers_orientation_distance(
                np.vstack(f1).T, np.vstack(f2).T)

    # Find pairs
    matches = []

    for k in range(min(spatial_dist.shape)):
        i, j = np.unravel_index(spatial_dist.argmin(), spatial_dist.shape)

        if spatial_dist[i, j] <= max_spatial_distance and \
           orientation_dist[i, j] <= max_orientation_distance:
            matches.append((l1[i], l2[j]))
            spatial_dist[i, :] = spatial_dist.max()
            spatial_dist[:, j] = spatial_dist.max()
        else:
            break

    return matches


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

        for p in f1:
            min_distance = np.linalg.norm(p - f2[0], ord=2)

            for q in f2[1:]:
                distance = np.linalg.norm(p - q, ord=2)

                if distance < min_distance:
                    min_distance = distance

            closest_distances.append(min_distance)

        return closest_distances

    closest_distances_f1 = _closest_distances(f1, f2)
    closest_distances_f2 = _closest_distances(f2, f1)

    return (max(np.mean(closest_distances_f1), np.mean(closest_distances_f2)),
            max(np.max(closest_distances_f1), np.max(closest_distances_f2)))


if __name__ == '__main__':
    import os
    import argparse
    from dfa import _utilities as _ut
    from skimage import io

    def _check_valid_path(path):
        """ Check for existing path (directory or file). """
        if not os.path.isdir(path) and not os.path.isfile(path):
            raise argparse.ArgumentTypeError('The given path is not a '
                                             'valid path!')

        return path

    def _check_float_0_1(variable):
        """ Check for floats in ]0, 1]. """
        try:
            variable = float(variable)
        except ValueError:
            raise argparse.ArgumentTypeError('The given variable cannot be '
                                             'converted to float!')

        if variable < 1e-10 or variable > 1:
            raise argparse.ArgumentTypeError('The given variable is out of the '
                                             'valid range (range is ]0, 1])')

        return variable

    def _check_positive_int(variable):
        """ Check for positive integers. """
        try:
            variable = int(variable)
        except ValueError:
            raise argparse.ArgumentTypeError('The given variable cannot be '
                                             'converted to int!')

        if variable <= 0:
            raise argparse.ArgumentTypeError('The given variable is out of '
                                             'the valid range (range is '
                                             ']0, +inf[).')

        return variable

    @_ut.static_vars(n=0, l=[])
    def _check_scales(variable):
        """ Check the scales validity. """
        try:
            variable = int(variable)
        except ValueError:
            raise argparse.ArgumentTypeError('The given variable cannot be '
                                             'converted to int!')

        if variable <= 0:
            raise argparse.ArgumentTypeError('The given variable is out of '
                                             'the valid range (range is '
                                             ']0, +inf[).')

        if _check_scales.n == 1 and variable < _check_scales.l[-1]:
            raise argparse.ArgumentTypeError('The second scale must be greater '
                                             'than the first one!')

        _check_scales.n += 1
        _check_scales.l.append(variable)

        return variable


    parser = argparse.ArgumentParser()

    group_images = parser.add_argument_group('Images')
    group_images.add_argument('input', type=_check_valid_path,
                              help='Path to input image.')

    group_detection = parser.add_argument_group('Detection')
    group_detection.add_argument('--fiber-sensitivity',
                                 type=_check_float_0_1, default=0.5,
                                 help='Sensitivity of detection to geometry in '
                                      'percentage (default is 0.5, valid range '
                                      'is ]0, 1]).')
    group_detection.add_argument('--intensity-sensitivity',
                                 type=_check_float_0_1, default=0.5,
                                 help='Sensitivity of detection to intensity in'
                                      ' percentage (default is 0.5, valid '
                                      'range is ]0, 1]).')
    group_detection.add_argument('--scales', type=_check_scales, nargs=3,
                                 default=[2, 4, 3],
                                 help='Scales to use in pixels (minimum, '
                                      'maximum, number of scales). Default is '
                                      '2 4 3.')

    group_reconstruction = parser.add_argument_group('Reconstruction')
    group_reconstruction.add_argument('--no-flat', action='store_true',
                                      help='Use non-flat structuring elements '
                                           'for fiber reconstruction (by '
                                           'default use flat structuring '
                                           'elements).')
    group_reconstruction.add_argument('--reconstruction-extent',
                                      type=_check_positive_int, default=20,
                                      help='Reconstruction extent in pixels '
                                           '(default is 20, range '
                                           'is ]0, +inf[).')

    group_medial = parser.add_argument_group('Medial axis')
    group_medial.add_argument('--smoothing', type=_check_positive_int,
                              default=20,
                              help='Smoothing of the output fibers '
                                   '(default is 20, range is is ]0, +inf[).')
    group_medial.add_argument('--fibers-minimal-length',
                              type=_check_positive_int, default=30,
                              help='Minimal length of a fiber in pixels '
                                   'default is 30, range is ]0, +inf[).')
    group_medial.add_argument('--output', type=_check_valid_path,
                              default=None,
                              help='Output path for saving detected fibers '
                                   '(default is None).')
    args = parser.parse_args()

    input_image = io.imread(args.input)

    if len(input_image.shape) == 2:
        fiber_image = input_image
    else:
        fiber_image = input_image[:, :, :, 1].mean(axis=0)

    coordinates = detect_fibers(
        fiber_image,
        scales=np.linspace(args.scales[0], args.scales[1],
                           int(args.scales[2])).tolist(),
        alpha=args.fiber_sensitivity,
        beta=1-args.intensity_sensitivity,
        length=args.reconstruction_extent,
        size=(args.scales[0]+args.scales[1])/2,
        smoothing=args.smoothing,
        min_length=args.fibers_minimal_length)

    # fiberness, directions = fiberness_filter(
    #     fiber_image,
    #     scales=np.linspace(args.scales[0], args.scales[1],
    #                        int(args.scales[2])).tolist(),
    #     alpha=args.fiber_sensitivity, beta=1-args.intensity_sensitivity)
    #
    # reconstructed_vesselness = reconstruct_fibers(
    #     fiberness, directions,
    #     length=args.reconstruction_extent,
    #     size=(args.scales[0]+args.scales[1])/2)
    #
    # coordinates = estimate_medial_axis(
    #     reconstructed_vesselness, smoothing=args.smoothing,
    #     min_length=args.fibers_minimal_length)

    if args.output is None:
        from matplotlib import pyplot as plt
        plt.imshow(fiber_image, cmap='gray', aspect='equal')
        for c in coordinates:
            plt.plot(*c, '-r')
        plt.show()
    else:
        _ut.write_points_to_txt(
            args.output,
            os.path.splitext(os.path.basename(args.input))[0],
            coordinates)

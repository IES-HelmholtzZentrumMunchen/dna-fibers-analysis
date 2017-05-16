"""
Detection module of the DNA fiber analysis package.

Use this module to detect fibers and extract their profiles.

Note that only quite straight fibers are detected for now.
"""
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev

from dfa import _scale_space_hessian as _sha
from dfa import _structuring_segments as _ss
from dfa import _grayscale_morphology as _gm
from dfa import _skeleton_pruning as _sk


def vesselness_filter(image, scales, alpha=0.5, beta=None, gamma=1):
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

    :param beta: Soft threshold of the intensity response (intensity-dependent).
    If not specified (None, default), the parameter is automatically estimated
    as proposed in Frangi et al.
    :type beta: positive float

    :param gamma: Gamma-normalization of the derivatives (see scale-space
    theory). If no scale is preferred, it should be set to 1 (default).
    :type gamma: positive float

    :return: The multiscale vesselness map and the corresponding vector field
    of the directions with the less intensity variation.
    :rtype: tuple of numpy.ndarray
    """
    vesselness = np.zeros((len(scales),) + image.shape)
    directions = np.zeros((len(scales), 2) + image.shape)

    for n, scale in enumerate(scales):
        hxx, hyy, hxy = _sha.single_scale_hessian(image, scale, gamma)
        (l1, l2), (v1, _) = _sha.hessian_eigen_decomposition(hxx, hyy, hxy)
        vesselness[n] = _sha.single_scale_vesselness(l1, l2, alpha, beta)
        directions[n] = v1

    ss, si, sj = vesselness.shape
    scale_selection = vesselness.argmax(axis=0)

    return (vesselness[scale_selection,
                       np.tile(np.arange(si).reshape(si, 1), (1, sj)),
                       np.tile(np.arange(sj).reshape(1, sj), (si, 1))],
            directions[scale_selection,
                       np.tile(np.arange(2).reshape(2, 1, 1), (1, si, sj)),
                       np.tile(np.arange(si).reshape(1, si, 1), (2, 1, sj)),
                       np.tile(np.arange(sj).reshape(1, 1, sj), (2, si, 1))])


def reconstruct_fibers(vesselness, directions, length, size):
    """
    Reconstruct fibers disconnections from vesselness map and directions of
    less intensity variations.

    The reconstruction is based on morphological closing using variant
    structuring segment. This algorithm is based on the following publication:
    Tankyevych et al. Curvilinear morpho-Hessian filter (2008). In: 5th IEEE
    International Symposium on Biomedical Imaging: From Nano to Macro, pp.
    1011-1014

    :param vesselness: Input vesselness map.
    :type vesselness: numpy.ndarray

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
        vesselness, directions, segments)
    segments = _ss.structuring_segments(regularized_directions, size, length, 0)

    # Reconstruct by morphological closing
    return _gm.adjunct_varying_closing(vesselness, segments)


def estimate_medial_axis(reconstruction, threshold):
    """
    Estimate the medial axis of the detected fibers from the reconstructed
    vesselness map.

    This current implementation uses a polynomial fitting to estimate,
    separately, the medial axis of the fibers.

    :param reconstruction: Reconstructed vesselness map.
    :type reconstruction: numpy.ndarray

    :param threshold: Threshold to use with the vesselness map.
    :type threshold: integer between 0 and 1

    :return: Coordinates of the medial axis lines of corresponding fibers.
    :rtype: list of tuples of float
    """
    # Threshold vesselness map and get connected components
    skeletons = skeletonize(reconstruction >= threshold)
    labels = label(skeletons)
    coordinates = []

    j, i = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))

    for l in range(1, labels.max() + 1):
        # fiber_skeleton = np.equal(labels, l)
        fiber_skeleton = _sk.prune_min(np.equal(labels, l))
        number_of_pixels = fiber_skeleton.sum()

        if number_of_pixels > 30:
            # we assume the skeleton has only one branch (it is pruned)
            splines, t = splprep(
                np.vstack((j[fiber_skeleton], i[fiber_skeleton])), s=5)
            t_sampled = np.linspace(t.min(), t.max(), number_of_pixels)
            x, y = splev(t_sampled, splines)
            coordinates.append((x, y))

    return coordinates, labels


if __name__ == '__main__':
    import argparse
    from skimage import io

    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--scales', type=float, nargs=3, default=[2, 4, 5],
                        help='Scales to use in pixels (minimum, maximum, '
                             'number of scales).')
    parser.add_argument('--no-flat', action='store_true',
                        help='Use non-flat structuring elements for fiber '
                             'reconstruction (by default use flat structuring '
                             'elements).')
    parser.add_argument('--reconstruction-extent', type=float, default=20,
                        help='Reconstruction extent in pixels (default is 20).')
    parser.add_argument('--vesselness-threshold', type=float, default=0.25,
                        help='Threshold used to binarize the vesselness map ('
                             'default is 0.25).')
    args = parser.parse_args()

    input_image = io.imread(args.input)

    plt.imshow(input_image, cmap='gray', aspect='equal')
    plt.show()

    vesselness, directions = vesselness_filter(
        input_image,
        scales=np.linspace(args.scales[0], args.scales[1],
                           int(args.scales[2])).tolist(),
        alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    plt.imshow(vesselness, cmap='gray', aspect='equal')
    plt.show()

    reconstructed_vesselness = reconstruct_fibers(
        vesselness, directions, args.reconstruction_extent,
        (args.scales[0]+args.scales[1])/2)

    plt.imshow(reconstructed_vesselness, cmap='gray', aspect='equal')
    plt.show()

    coordinates, labels = estimate_medial_axis(reconstructed_vesselness,
                                               args.vesselness_threshold)

    plt.imshow(labels, cmap='gray', aspect='equal')
    plt.show()

    plt.imshow(input_image, cmap='gray', aspect='equal')
    for c in coordinates:
        plt.plot(*c, '-r')
    plt.show()

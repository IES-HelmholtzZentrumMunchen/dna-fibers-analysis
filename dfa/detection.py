"""
Detection module of the DNA fiber analysis package.

Use this module to detect fibers and extract their profiles.

Note that only quite straight fibers are detected for now.
"""
import numpy as np
from skimage.measure import label

from dfa import _scale_space_hessian as _sha
from dfa import _structuring_segments as _ss
from dfa import _grayscale_morphology as _gm


def vesselness_filter(image, scales, alpha, beta):
    """
    Enhance fiber image using a multi-scale vesselness filter.

    This vesselness is based on the Frangi filter (Frangi et al. Multiscale
    vessel enhancement filtering (1998). In: Medical ImageComputing and
    Computer-Assisted Intervention, vol. 1496, pp. 130-137).

    :param image: Input image to filter.
    :type image: numpy.ndarray

    :param scales: Sizes range.
    :type scales: list of float or iterable of float

    :param alpha: Soft threshold of the tubular shape weighting term.
    :type alpha: float between 0 and 1

    :param beta: Soft threshold of the intensity response.
    :type beta: float between 0 and 1

    :return: The multiscale vesselness map and the corresponding vector field
    of the directions with the less intensity variation.
    :rtype: tuple of numpy.ndarray
    """
    vesselness = np.zeros((len(scales),) + image.shape)
    directions = np.zeros((len(scales), 2) + image.shape)

    for n, scale in enumerate(scales):
        hxx, hyy, hxy = _sha.single_scale_hessian(image, scale)
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
    regularized_directions = np.zeros(directions.shape)
    regularized_directions[0] = _gm.adjunct_varying_dilation(
        directions[0], segments)
    regularized_directions[1] = _gm.adjunct_varying_dilation(
        directions[1], segments)
    segments = _ss.structuring_segments(regularized_directions, size, length, 0)

    # Reconstruct by morphological closing
    return _gm.adjunct_varying_closing(vesselness, segments)


def estimate_medial_axis(reconstruction, threshold, degree=3):
    """
    Estimate the medial axis of the detected fibers from the reconstructed
    vesselness map.

    This current implementation uses a polynomial fitting to estimate,
    separately, the medial axis of the fibers.

    :param reconstruction: Reconstructed vesselness map.
    :type reconstruction: numpy.ndarray

    :param threshold: Threshold to use with the vesselness map.
    :type threshold: integer between 0 and 1

    :param degree: Degree of the fitted polynomial.
    :type degree: positive int

    :return: Coordinates of the medial axis lines of corresponding fibers.
    :rtype: list of tuples of float
    """
    # Threshold vesselness map and get connected components
    labels = label(reconstruction >= threshold)
    coordinates = []

    for l in range(1, labels.max() + 1):
        # Get points coordinate for current component
        j, i = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))
        x = j[labels == l]
        x = x.reshape(x.size, 1)
        y = i[labels == l]
        y = y.reshape(y.size, 1)

        # Fit a polynome with least-squares
        xm = np.ones((x.size, 1))
        for d in range(1, degree + 1):
            xm = np.hstack((xm, np.power(x, d)))
        b = np.dot(np.dot(np.linalg.inv(np.dot(xm.T, xm)), xm.T), y)

        # Get medial axis coordinates
        x = np.linspace(x.min(), x.max())
        coordinates.append(
            (x, (np.tile(b, (1, x.size)) *
                 np.power(np.tile(x, (b.size, 1)),
                          np.tile(np.arange(b.size).reshape(b.size, 1),
                                  (1, x.size)))).sum(axis=0)))

    return coordinates


if __name__ == '__main__':
    import argparse
    from skimage import io

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--scales', type=float, nargs=3, default=[3, 5, 5],
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
    parser.add_argument('--polynomial-degree', type=int, default=3,
                        help='The degree of the polynomial fitted to estimate '
                             'the medial axis of the fibers (default is 3).')
    args = parser.parse_args()

    input_image = io.imread(args.input)

    vesselness, directions = vesselness_filter(
        input_image,
        np.linspace(args.scales[0], args.scales[1],
                    int(args.scales[2])).tolist(),
        alpha=args.alpha, beta=args.beta)

    reconstructed_vesselness = reconstruct_fibers(
        vesselness, directions, args.reconstruction_extent,
        (args.scales[0]+args.scales[1])/2)

    coordinates = estimate_medial_axis(reconstructed_vesselness,
                                       args.vesselness_threshold,
                                       args.polynomial_degree)

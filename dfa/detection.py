"""
Detection module of the DNA fiber analysis package.

Use this module to detect fibers and extract their profiles.

Note that only quite straight fibers are detected for now.
"""
import numpy as np

import _scale_space_hessian as _sha


def vesselness_filter(image, scales, alpha, beta):
    """
    Enhance fiber image using a multiscale vesselness filter.

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


def reconstruct_fibers(vesselness, principal_directions, length):
    pass  # TODO: implement fibers reconstruction


def estimate_medial_axis(reconstruction):
    pass  # TODO: implement medial axis estimation


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
    args = parser.parse_args()

    input_image = io.imread(args.input)

    vesselness, directions = vesselness_filter(
        input_image,
        np.linspace(args.scales[0], args.scales[1],
                    int(args.scales[2])).tolist(),
        alpha=args.alpha, beta=args.beta)

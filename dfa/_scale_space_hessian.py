"""
Module for hessian analysis within scale-space theory.
"""
import numpy as np
from scipy.signal import convolve2d


def discrete_centered_space(k):
    """
    Compute a centered discrete sequence (1D) from its half-size.

    :param k: Half size of the sequence. The final size of the sequence will be
    equal to 2*k+1 elements.
    :type k: strictly positive int

    :return: A centered discrete sequence.
    :rtype: numpy.ndarray
    """
    return np.linspace(-k, k, 2*k+1).reshape(2*k+1, 1)


def gaussian_kernel(sigma, k):
    """
    Compute a discrete sequence of size 2*k+1 corresponding to a centered
    gaussian kernel of width sigma.

    :param sigma: Width of the gaussian kernel.
    :type sigma: strictly positive float

    :param k: Half size of the sequence. The final size of the sequence will be
    equal to 2*k+1 elements.
    :type k: strictly positive int

    :return: A discrete sequence of a gaussian kernel.
    :rtype: numpy.ndarray
    """
    x = discrete_centered_space(k)
    v = 1/(sigma * np.sqrt(2 * np.pi)) * \
        np.exp(-0.5 * np.power(x, 2) / sigma ** 2)

    return v


def gaussian_first_derivative_kernel(sigma, k):
    """
    Compute a discrete sequence of size 2*k+1 corresponding to the first order
    derivative of a centered gaussian kernel of width sigma.

    :param sigma: Width of the gaussian kernel.
    :type sigma: strictly positive float

    :param k: Half size of the sequence. The final size of the sequence will be
    equal to 2*k+1 elements.
    :type k: strictly positive int

    :return: A discrete sequence of the first order derivative of a
    gaussian kernel.
    :rtype: numpy.ndarray
    """
    x = discrete_centered_space(k)
    v = -x/(sigma ** 3 * np.sqrt(2 * np.pi)) * \
        np.exp(-0.5 * np.power(x, 2) / sigma ** 2)

    return v


def gaussian_second_derivative_kernel(sigma, k):
    """
    Compute a discrete sequence of size 2*k+1 corresponding to the second order
    derivative of a centered gaussian kernel of width sigma.

    :param sigma: Width of the gaussian kernel.
    :type sigma: strictly positive float

    :param k: Half size of the sequence. The final size of the sequence will be
    equal to 2*k+1 elements.
    :type k: strictly positive int

    :return: A discrete sequence of the second order derivative of a
    gaussian kernel.
    :rtype: numpy.ndarray
    """
    x = discrete_centered_space(k)
    v = -(sigma ** 2 - np.power(x, 2)) / (sigma ** 5 * np.sqrt(2 * np.pi)) * \
        np.exp(-0.5 * np.power(x, 2) / sigma ** 2)

    return v


def single_scale_hessian(image, size, gamma=1):
    """
    Compute the Hessian matrix for an image using the scale-space theory.

    :param image: Input image.
    :type image: np.ndarray

    :param size: Size of the current scale-space.
    :type size: strictly positive float

    :param gamma: Gamma-normalization of the derivatives (see scale-space
    theory). If no scale is preferred, it should be set to 1 (default).
    :type gamma: positive float

    :return: The Hessian matrix elements for input image (hxx, hyy, hxy).
    :rtype: tuple of numpy.ndarray
    """
    k = round(6*size//2)

    hxx = convolve2d(image,
                     np.multiply(gaussian_second_derivative_kernel(size, k).T,
                                 gaussian_kernel(size, k)),
                     mode='same')
    hyy = convolve2d(image,
                     np.multiply(gaussian_second_derivative_kernel(size, k),
                                 gaussian_kernel(size, k).T),
                     mode='same')
    hxy = convolve2d(image,
                     np.multiply(gaussian_first_derivative_kernel(size, k),
                                 gaussian_first_derivative_kernel(size, k).T),
                     mode='same')

    factor = size ** gamma  # size ** (2*gamma/2)

    return factor * hxx, factor * hyy, factor * hxy


def hessian_eigen_decomposition(hxx, hyy, hxy):
    """
    Compute the eigen values/vectors for Hessian matrix.

    :param hxx: Hessian coefficient for second order derivative in X.
    :type hxx: numpy.ndarray

    :param hyy: Hessian coefficient for second order derivative in Y.
    :type hyy: numpy.ndarray

    :param hxy: Hessian coefficient for first order derivative in X and Y.
    :type hxy: numpy.ndarray

    :return: The eigen values and the eigen vectors ((l1, l2), (v1, v2)).
    :rtype: tuple of tuples of numpy.ndarray
    """
    trace = hxx + hyy
    determinant = hxx * hyy - hxy * hxy

    # Compute eigen-values
    tmp = np.sqrt(np.power(trace, 2) - 4 * determinant)
    l1 = (trace + tmp) / 2
    l2 = (trace - tmp) / 2

    # Order by ascending absolute values
    swap = np.abs(l1) > np.abs(l2)
    tmp = l1[swap]
    l1[swap] = l2[swap]
    l2[swap] = tmp

    # Compute eigen-vectors
    v1 = np.zeros((2, hxx.shape[0], hxx.shape[1]))
    v2 = np.zeros(v1.shape)

    hxy_is_zero = np.abs(hxy) < 1e-10
    hxy_is_not_zero = np.bitwise_not(hxy_is_zero)
    hxx_greater_hyy = np.abs(hxx) > np.abs(hyy)
    hxx_lesser_or_equal_hyy = np.bitwise_not(hxx_greater_hyy)
    hxy_is_zero_and_hxx_is_greater = np.bitwise_and(hxy_is_zero,
                                                    hxx_greater_hyy)
    hxy_is_zero_and_hxx_is_lesser = np.bitwise_and(hxy_is_zero,
                                                   hxx_lesser_or_equal_hyy)

    v1[0, hxy_is_zero_and_hxx_is_greater] = 1
    v1[1, hxy_is_zero_and_hxx_is_greater] = 0
    v2[0, hxy_is_zero_and_hxx_is_greater] = 0
    v2[1, hxy_is_zero_and_hxx_is_greater] = 1

    v1[0, hxy_is_zero_and_hxx_is_lesser] = 0
    v1[1, hxy_is_zero_and_hxx_is_lesser] = 1
    v2[0, hxy_is_zero_and_hxx_is_lesser] = 1
    v2[1, hxy_is_zero_and_hxx_is_lesser] = 0

    v1[0, hxy_is_not_zero] = l1[hxy_is_not_zero] - hxx[hxy_is_not_zero]
    v1[1, hxy_is_not_zero] = hxy[hxy_is_not_zero]
    v2[0, hxy_is_not_zero] = l2[hxy_is_not_zero] - hxx[hxy_is_not_zero]
    v2[1, hxy_is_not_zero] = hxy[hxy_is_not_zero]

    return (l1, l2), (v1, v2)


def single_scale_vesselness(l1, l2, alpha=0.5, beta=0.5):
    """
    Compute the vesselness using the scale-space theory.

    This vesselness is based on the Frangi filter (Frangi et al. Multi-scale
    vessel enhancement filtering (1998). In: Medical ImageComputing and
    Computer-Assisted Intervention, vol. 1496, pp. 130-137).

    :param l1: First eigen-value of the Hessian matrix of an input image
    for a given scale.
    :type l1: numpy.ndarray

    :param l2: First eigen-value of the Hessian matrix of an input image
    for a given scale.
    :type l2: numpy.ndarray

    :param alpha: Soft threshold of the tubular shape weighting term. Default is
    the recommended value (i.e. 0.5).
    :type alpha: float between 0 and 1

    :param beta: Soft threshold of the intensity response (intensity-dependent)
    as a percentage of maximal Hessian norm (as proposed in Frangi et al.).
    :type beta: positive float

    :return: The vesselness map.
    :rtype: numpy.ndarray
    """
    l2_is_negative = l2 < 0

    rb = np.zeros(l1.shape)
    rb[l2_is_negative] = np.abs(l1[l2_is_negative]) / np.abs(l2[l2_is_negative])

    s = np.zeros(l1.shape)
    s[l2_is_negative] = np.sqrt(
        np.power(l1[l2_is_negative], 2) + np.power(l2[l2_is_negative], 2))

    beta *= s.max()

    vesselness = np.zeros(l1.shape)
    vesselness[l2_is_negative] = np.exp(
        -0.5 * np.power(rb[l2_is_negative] / alpha, 2)) * \
        (1 - np.exp(-0.5 * np.power(s[l2_is_negative] / beta, 2)))

    return vesselness

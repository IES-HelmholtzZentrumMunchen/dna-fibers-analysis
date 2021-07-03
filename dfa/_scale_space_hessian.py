"""
Module for hessian analysis within scale-space theory.
"""
import numpy as np
from scipy.signal import fftconvolve


def discrete_centered_space(k):
    """
    Compute a centered discrete sequence (1D) from its half-size.

    Parameters
    ----------
    k : strictly positive int
        Half size of the sequence. The final size of the sequence will be equal
        to 2*k+1 elements.

    Returns
    -------
    numpy.ndarray
        A centered discrete sequence.
    """
    return np.linspace(-k, k, 2*k+1).reshape(2*k+1, 1)


def gaussian_kernel(sigma, k):
    """
    Compute a discrete gaussian kernel.

    Compute a discrete sequence of size 2*k+1 corresponding to a centered
    gaussian kernel of width sigma.

    Parameters
    ----------
    sigma : strictly positive float
        Width of the gaussian kernel.

    k : strictly positive int
        Half size of the sequence. The final size of the sequence will be equal
        to 2*k+1 elements.

    Returns
    -------
    numpy.ndarray
        A discrete sequence of a gaussian kernel.

    See Also
    --------
    discrete_centered_space : linear centered discrete kernel.
    gaussian_first_derivative_kernel : discrete kernel of first order Gaussian
        derivative.
    gaussian_second_derivative_kernel : discrete kernel of second order Gaussian
        derivative.
    """
    x = discrete_centered_space(k)
    return 1/(sigma * np.sqrt(2 * np.pi)) * \
        np.exp(-0.5 * np.power(x, 2) / sigma ** 2)


def gaussian_first_derivative_kernel(sigma, k):
    """
    Compute a discrete kernel of the first Gaussian derivative.

    Compute a discrete sequence of size 2*k+1 corresponding to the first order
    derivative of a centered gaussian kernel of width sigma.

    Parameters
    ----------
    sigma : strictly positive float
        Width of the gaussian kernel.

    k : strictly positive int
        Half size of the sequence. The final size of the sequence will be equal
        to 2*k+1 elements.

    Returns
    -------
    numpy.ndarray
        A discrete sequence of the first order derivative of a gaussian kernel.

    See Also
    --------
    discrete_centered_space : linear centered discrete kernel.
    gaussian_kernel : discrete Gaussian kernel.
    gaussian_second_derivative_kernel : discrete kernel of second order Gaussian
        derivative.
    """
    x = discrete_centered_space(k)
    return -x/(sigma ** 3 * np.sqrt(2 * np.pi)) * \
        np.exp(-0.5 * np.power(x, 2) / sigma ** 2)


def gaussian_second_derivative_kernel(sigma, k):
    """
    Compute a discrete kernel of the second Gaussian derivative.

    Compute a discrete sequence of size 2*k+1 corresponding to the second order
    derivative of a centered gaussian kernel of width sigma.

    Parameters
    ----------
    sigma : strictly positive float
        Width of the gaussian kernel.

    k : strictly positive int
        Half size of the sequence. The final size of the sequence will be equal
        to 2*k+1 elements.

    Returns
    -------
    numpy.ndarray
        A discrete sequence of the second order derivative of a gaussian kernel.

    See Also
    --------
    discrete_centered_space : linear centered discrete kernel.
    gaussian_first_derivative_kernel : discrete kernel of first order Gaussian
        derivative.
    gaussian_second_derivative_kernel : discrete kernel of second order Gaussian
        derivative.
    """
    x = discrete_centered_space(k)
    return -(sigma ** 2 - np.power(x, 2)) / (sigma ** 5 * np.sqrt(2 * np.pi)) * \
        np.exp(-0.5 * np.power(x, 2) / sigma ** 2)


def single_scale_hessian(image, size, gamma=1):
    """
    Compute the Hessian matrix for an image using the scale-space theory.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    size : strictly positive float
        Size of the current scale-space.

    gamma : positive float
        Gamma-normalization of the derivatives (see scale-space theory). If no
        scale is preferred, it should be set to 1 (default).

    Returns
    -------
    tuple of numpy.ndarray
        The Hessian matrix elements for input image (hxx, hyy, hxy).

    See Also
    --------
    hessian_eigen_decomposition : Eigen-decomposition of Hessian matrix.
    """
    k = round(6*size//2)

    g0 = gaussian_kernel(size, k)
    g1 = gaussian_first_derivative_kernel(size, k)
    g2 = gaussian_second_derivative_kernel(size, k)

    hxx = fftconvolve(image, np.multiply(g2.T, g0), mode='same')
    hyy = fftconvolve(image, np.multiply(g2, g0.T), mode='same')
    hxy = fftconvolve(image, np.multiply(g1, g1.T), mode='same')

    factor = size ** gamma  # size ** (2*gamma/2)

    return factor * hxx, factor * hyy, factor * hxy


def hessian_eigen_decomposition(hxx, hyy, hxy):
    """
    Compute the eigen values/vectors for Hessian matrix.

    Parameters
    ----------
    hxx : numpy.ndarray
        Hessian coefficient for second order derivative in X.

    hyy : numpy.ndarray
        Hessian coefficient for second order derivative in Y.

    hxy : numpy.ndarray
        Hessian coefficient for first order derivative in X and Y.

    Returns
    -------
    tuple of tuples of numpy.ndarray
        The eigen values and the eigen vectors ((l1, l2), (v1, v2)).

    See Also
    --------
    single_scale_hessian : estimation of Hessian matrix
    single_scale_vesselness : single-scale Frangi filter
    structurness_parameter_auto : automatic estimation of the intensity drop
        for the Frangi filter (single_scale_vesselness).
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


# noinspection SpellCheckingInspection
def single_scale_vesselness(l1, l2, mask=None, alpha=0.5, beta=1.0):
    """
    Compute the vesselness using the scale-space theory.

    This vesselness is based on the Frangi filter [1]_.

    Parameters
    ----------
    l1 : numpy.ndarray
        First eigen-value of the Hessian matrix of an input image for a
        given scale.

    l2 : numpy.ndarray
        First eigen-value of the Hessian matrix of an input image for a
        given scale.

    mask : numpy.ndarray | None
        Mask within which the vesselness map will be computed.

    alpha : float between 0 and 1
        Soft threshold of the tubular shape weighting term. Default is the
        recommended value (i.e. 0.5).

    beta : positive float
        Soft threshold of the intensity response (intensity dynamic-dependent)
        as a percentage of an automatically estimated parameter. Default is 1.0,
        i.e. the parameter as it is automatically estimated.

    Returns
    -------
    numpy.ndarray
        The vesselness map.

    See Also
    --------
    hessian_eigen_decomposition : Eigen-decomposition of Hessian matrix.
    structurness_parameter_auto : automatic estimation of the intensity drop
        for the Frangi filter.

    Notes
    -----
    .. [1] Frangi et al. (1998) Multi-scale vessel enhancement filtering. In:
       *Medical Image Computing and Computer-Assisted Intervention*, vol. 1496,
       pp. 130-137.
    """
    l2_is_negative = l2 < 0

    rb = np.zeros(l1.shape)
    rb[l2_is_negative] = np.abs(l1[l2_is_negative]) / np.abs(l2[l2_is_negative])

    s = np.zeros(l1.shape)
    s[l2_is_negative] = np.sqrt(
        np.power(l1[l2_is_negative], 2) + np.power(l2[l2_is_negative], 2))

    beta *= structurness_parameter_auto(s, mask)

    vesselness = np.zeros(l1.shape)
    vesselness[l2_is_negative] = np.exp(
        -0.5 * np.power(rb[l2_is_negative] / alpha, 2)) * \
        (1 - np.exp(-0.5 * np.power(s[l2_is_negative] / beta, 2)))

    return vesselness


def structurness_parameter_auto(structurness, mask=None, res=128):
    """
    Automatically estimate the structurness drop parameter from the
    structurness image.

    The parameter is estimated using an histogram-based method: the optimal
    point value is calculated geometrically, by the triangle thresholding
    method [2]_. The estimated point is the point that has the
    largest distance to the line drawn between the mode and the maximal value.

    Note that the zero values are not taken into account when computing
    the histogram.

    Parameters
    ----------
    structurness : numpy.ndarray
        Structurness image measured from Hessian tensors.

    mask : numpy.ndarray | None
        Mask within which the structurness weight will be computed.

    res : int
        Resolution used for computing the histogram (number of bins).

    Returns
    -------
    float
        The estimated drop parameter.

    See Also
    --------
    hessian_eigen_decomposition : Eigen-decomposition of Hessian matrix.
    single_scale_vesselness : single-scale Frangi filter

    Notes
    -----
    .. [2] Zack GW, Rogers WE, Latt SA (1977) Automatic measurement of sister
       chromatid exchange frequency. In: *J. Histochem. Cytochem.*, vol. 25,
       num. 7, pp. 741–53.
    """
    if mask is not None:
        hy, hx = np.histogram(
            structurness[np.bitwise_and(structurness > 0, mask)], res)
    else:
        hy, hx = np.histogram(structurness[structurness > 0], res)

    hx = hx[:-1] + np.diff(hx)

    i = hy.argmax()
    ax, ay = hx[i], hy[i]
    bx, by = hx[-1], hy[-1]

    slope = (by - ay) / (bx - ax)
    intercept = ay - slope * ax

    vx = bx - ax
    vy = by - ay

    max_distance = 0
    parameter = 0
    for x, y in zip(hx[i+1:-1], hy[i+1:-1]):
        cx = (vx * x + vy * y - vy * intercept) / (vx + vy * slope)
        cy = slope * cx + intercept
        distance = np.sqrt((cx - x)**2 + (cy - y)**2)

        if distance > max_distance:
            max_distance = distance
            parameter = x

    return parameter

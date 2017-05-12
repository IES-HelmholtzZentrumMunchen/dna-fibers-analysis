"""
Module for grayscale morphology operations with varying structuring elements,
used for reconstruction during the fibers detection.
"""
import numpy as np


def varying_filtering_2d(image, structuring_elements, function_map,
                         function_reduce):
    """
    Do spatial filtering in 2D with (possibly) varying structuring elements.

    Since it is vectorized, it is very fast

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param function_map: Function of the filter domain used when mapping.
    :type function_map: function

    :param function_reduce: Function of the filter domain used when reducing.
    :type function_reduce: function

    :return: The filtered image.
    :rtype: numpy.ndarray
    """
    filtered = np.zeros(image.shape)

    ki = structuring_elements[0, 0].shape[0] // 2
    kj = structuring_elements[0, 0].shape[1] // 2

    oj, oi = np.meshgrid(range(-kj, kj + 1), range(-ki, ki + 1))

    for i in range(ki, image.shape[0] - ki):
        for j in range(kj, image.shape[1] - kj):
            filtered[i, j] = function_reduce(
                function_map(image[i + oi, j + oj],
                             structuring_elements[i, j]))

    return filtered


def varying_dilation(image, structuring_elements):
    """
    Compute a grayscale dilation with a (possibly) varying structuring element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :return: The dilated image.
    :rtype: numpy.ndarray
    """
    return varying_filtering_2d(image, structuring_elements, np.add, np.max)


def varying_erosion(image, structuring_elements):
    """
    Compute a grayscale erosion with a (possibly) varying structuring  element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :return: The eroded image.
    :rtype: numpy.ndarray
    """
    return varying_filtering_2d(image, structuring_elements,
                                np.subtract, np.min)


def varying_opening(image, structuring_elements):
    """
    Compute a grayscale opening with a (possibly) varying structuring  element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :return: The opened image.
    :rtype: numpy.ndarray
    """
    return varying_dilation(varying_erosion(image, structuring_elements),
                            structuring_elements)


def varying_closing(image, structuring_elements):
    """
    Compute a grayscale closing with a (possibly) varying structuring  element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :return: The closed image.
    :rtype: numpy.ndarray
    """
    return varying_erosion(varying_dilation(image, structuring_elements),
                           structuring_elements)


def adjunct_varying_filtering_2d(image, structuring_elements, function_map,
                                 function_reduce, initialization):
    """
    Compute the grayscale adjunct filtering with a (possibly) varying
    structuring element.

    This is particularly important when using varying structuring elements, as
    the standard operators do not take into account the varying elements.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param function_map: Function of the filter domain used when mapping.
    :type function_map: function

    :param function_reduce: Function of the filter domain used when reducing.
    :type function_reduce: function

    :param initialization: Initialization element.
    :type initialization: float or int

    :return: The filtered image with the adjunct operator.
    :rtype: numpy.ndarray
    """
    filtered = np.zeros(image.shape)
    filtered[:] = initialization

    ki = structuring_elements[0, 0].shape[0] // 2
    kj = structuring_elements[0, 0].shape[1] // 2

    oj, oi = np.meshgrid(range(-kj, kj + 1), range(-ki, ki + 1))

    for i in range(ki, image.shape[0] - ki):
        for j in range(kj, image.shape[1] - kj):
            filtered[i + oi, j + oj] = function_reduce(
                filtered[i + oi, j + oj],
                function_map(image[i, j], structuring_elements[i, j]))

    filtered[filtered == initialization] = 0

    return filtered


def adjunct_varying_dilation(image, structuring_elements):
    """
    Compute the grayscale adjunct dilation to the erosion with a (possibly)
    varying structuring element.

    This is particularly important when using varying structuring elements, as
    the standard operators do not take into account the varying elements.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :return: The dilated image with the adjunct of the erosion.
    :rtype: numpy.ndarray
    """
    return adjunct_varying_filtering_2d(image, structuring_elements,
                                        np.add, np.maximum, -np.inf)


def adjunct_varying_erosion(image, structuring_elements):
    """
    Compute the grayscale adjunct erosion to the dilation with a (possibly)
    varying structuring element.

    This is particularly important when using varying structuring elements, as
    the standard operators do not take into account the varying elements.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :return: The eroded image with the adjunct of the dilation.
    :rtype: numpy.ndarray
    """
    return adjunct_varying_filtering_2d(image, structuring_elements,
                                        np.subtract, np.minimum, np.inf)


def adjunct_varying_opening(image, structuring_elements, adjunct_dilation=True):
    """
    Compute a grayscale opening with a (possibly) varying structuring  element
    using adjunct operator.

    Use of adjunct operator is particularly important when using varying
    structuring elements, as the standard operators do not take into account
    the varying elements.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param adjunct_dilation: Use adjunct of erosion as dilation (True by
    default).
    :type adjunct_dilation: bool

    :return: The opened image.
    :rtype: numpy.ndarray
    """
    if adjunct_dilation:
        return adjunct_varying_dilation(
            varying_erosion(image, structuring_elements), structuring_elements)
    else:
        return varying_dilation(
            adjunct_varying_erosion(image, structuring_elements),
            structuring_elements)


def adjunct_varying_closing(image, structuring_elements, adjunct_dilation=True):
    """
    Compute a grayscale closing with a (possibly) varying structuring  element
    using adjunct operator..

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param adjunct_dilation: Use adjunct of erosion as dilation (True by
    default).
    :type adjunct_dilation: bool

    :return: The closed image.
    :rtype: numpy.ndarray
    """
    if adjunct_dilation:
        return varying_erosion(
            adjunct_varying_dilation(image, structuring_elements),
            structuring_elements)
    else:
        return adjunct_varying_erosion(
            varying_dilation(image, structuring_elements), structuring_elements)

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
    :type structuring_elements: numpy.ndarray

    :param function_map: Function of the filter domain used when mapping.
    :type function_map: function

    :param function_reduce: Function of the filter domain used when reducing.
    :type function_reduce: function

    :return: The filtered image.
    :rtype: numpy.ndarray
    """
    w = structuring_elements.shape[2]
    t = w // 2
    filtered = np.zeros((structuring_elements[0, 0, :].size,) + image.shape)

    k = np.array(range(filtered.shape[0]))
    i, _, j = np.meshgrid(range(t, image.shape[0] - t), k,
                          range(t, image.shape[1] - t))

    bi, bj = k // w, k % w
    bi = np.tile(bi.reshape(bi.size, 1, 1), (1, i.shape[1], i.shape[2]))
    bj = np.tile(bj.reshape(bj.size, 1, 1), (1, i.shape[1], i.shape[2]))

    filtered[:, t:image.shape[0] - t, t:image.shape[1] - t] = function_map(
        image[i + bi - t, j + bj - t], structuring_elements[i, j, bi, bj])

    return function_reduce(filtered, axis=0)


def varying_dilation(image, structuring_elements):
    """
    Compute a grayscale dilation with a (possibly) varying structuring element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: numpy.ndarray

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
    :type structuring_elements: numpy.ndarray

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
    :type structuring_elements: numpy.ndarray

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
    :type structuring_elements: numpy.ndarray

    :return: The closed image.
    :rtype: numpy.ndarray
    """
    return varying_erosion(varying_dilation(image, structuring_elements),
                           structuring_elements)

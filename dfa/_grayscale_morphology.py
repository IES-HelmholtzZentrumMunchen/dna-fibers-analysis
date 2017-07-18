"""
Module for grayscale morphology operations with varying structuring elements,
used for reconstruction during the fibers detection.
"""
import numpy as np


def varying_filtering_2d(image, structuring_elements, function_map,
                         function_reduce, mask):
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

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

    :return: The filtered image.
    :rtype: numpy.ndarray
    """
    filtered = np.zeros(image.shape)

    ki = structuring_elements[0, 0].shape[0] // 2
    kj = structuring_elements[0, 0].shape[1] // 2

    oj, oi = np.meshgrid(range(-kj, kj + 1), range(-ki, ki + 1))

    for i in range(ki, image.shape[0] - ki):
        for j in range(kj, image.shape[1] - kj):
            if mask[i, j]:
                filtered[i, j] = function_reduce(
                    function_map(image[i + oi, j + oj],
                                 structuring_elements[i, j]))

    return filtered


def varying_dilation(image, structuring_elements, mask):
    """
    Compute a grayscale dilation with a (possibly) varying structuring element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

    :return: The dilated image.
    :rtype: numpy.ndarray
    """
    return varying_filtering_2d(image, structuring_elements,
                                np.add, np.max, mask)


def varying_erosion(image, structuring_elements, mask):
    """
    Compute a grayscale erosion with a (possibly) varying structuring  element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

    :return: The eroded image.
    :rtype: numpy.ndarray
    """
    return varying_filtering_2d(image, structuring_elements,
                                np.subtract, np.min, mask)


def varying_opening(image, structuring_elements, mask):
    """
    Compute a grayscale opening with a (possibly) varying structuring  element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

    :return: The opened image.
    :rtype: numpy.ndarray
    """
    return varying_dilation(varying_erosion(image, structuring_elements, mask),
                            structuring_elements, mask)


def varying_closing(image, structuring_elements, mask):
    """
    Compute a grayscale closing with a (possibly) varying structuring  element.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

    :return: The closed image.
    :rtype: numpy.ndarray
    """
    return varying_erosion(varying_dilation(image, structuring_elements, mask),
                           structuring_elements, mask)


def adjunct_varying_filtering_2d(image, structuring_elements, function_map,
                                 function_reduce, initialization, mask):
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

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

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
            if mask[i, j]:
                filtered[i + oi, j + oj] = function_reduce(
                    filtered[i + oi, j + oj],
                    function_map(image[i, j], structuring_elements[i, j]))

    filtered[filtered == initialization] = 0

    return filtered


def adjunct_varying_dilation(image, structuring_elements, mask):
    """
    Compute the grayscale adjunct dilation to the erosion with a (possibly)
    varying structuring element.

    This is particularly important when using varying structuring elements, as
    the standard operators do not take into account the varying elements.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

    :return: The dilated image with the adjunct of the erosion.
    :rtype: numpy.ndarray
    """
    return adjunct_varying_filtering_2d(image, structuring_elements,
                                        np.add, np.maximum, -np.inf, mask)


def adjunct_varying_erosion(image, structuring_elements, mask):
    """
    Compute the grayscale adjunct erosion to the dilation with a (possibly)
    varying structuring element.

    This is particularly important when using varying structuring elements, as
    the standard operators do not take into account the varying elements.

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: The operations will be processed only inside the mask.
    :type mask: numpy.ndarray

    :return: The eroded image with the adjunct of the dilation.
    :rtype: numpy.ndarray
    """
    return adjunct_varying_filtering_2d(image, structuring_elements,
                                        np.subtract, np.minimum, np.inf, mask)


def adjunct_varying_opening(image, structuring_elements, mask=None,
                            extent_mask=None, adjunct_dilation=True):
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

    :param mask: Mask of where could be object of interest (default is the
    whole image when set to None).
    :type mask: numpy.ndarray

    :param extent_mask: Mask of where the structuring elements could reach
    (default is the whole image when set to None). It should contain the mask.
    :type extent_mask: numpy.ndarray

    :param adjunct_dilation: Use adjunct of erosion as dilation (True by
    default).
    :type adjunct_dilation: bool

    :return: The opened image.
    :rtype: numpy.ndarray
    """
    if mask is None:
        mask = np.ones(image.shape).astype(bool)

    if extent_mask is None:
        extent_mask = mask.copy()

    if adjunct_dilation:
        return adjunct_varying_dilation(
            varying_erosion(image, structuring_elements, extent_mask),
            structuring_elements, mask)
    else:
        return varying_dilation(
            adjunct_varying_erosion(image, structuring_elements, mask),
            structuring_elements, extent_mask)


def adjunct_varying_closing(image, structuring_elements, mask=None,
                            extent_mask=None, adjunct_dilation=True):
    """
    Compute a grayscale closing with a (possibly) varying structuring  element
    using adjunct operator..

    :param image: Image to filter.
    :type image: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: Mask of where could be object of interest (default is the
    whole image when set to None).
    :type mask: numpy.ndarray

    :param extent_mask: Mask of where the structuring elements could reach
    (default is the whole image when set to None). It should contain the mask.
    :type extent_mask: numpy.ndarray

    :param adjunct_dilation: Use adjunct of erosion as dilation (True by
    default).
    :type adjunct_dilation: bool

    :return: The closed image.
    :rtype: numpy.ndarray
    """
    if mask is None:
        mask = np.ones(image.shape).astype(bool)

    if extent_mask is None:
        extent_mask = mask.copy()

    if adjunct_dilation:
        return varying_erosion(
            adjunct_varying_dilation(image, structuring_elements, mask),
            structuring_elements, extent_mask)
    else:
        return adjunct_varying_erosion(
            varying_dilation(image, structuring_elements, extent_mask),
            structuring_elements, mask)


def morphological_regularization(image, directions, structuring_elements,
                                 mask=None):
    """
    Regularize the directions field using guided morphological dilation.

    It computes grayscale adjunct filtering of directions field, guided with an
    image and with a (possibly) varying structuring element.

    This is particularly important when using varying structuring elements, as
    the standard operators do not take into account the varying elements.

    :param image: Image used as guide.
    :type image: numpy.ndarray

    :param directions: The directions to regularize.
    :type directions: numpy.ndarray

    :param structuring_elements: Structuring elements possibly varying.
    :type structuring_elements: dict

    :param mask: Mask to limit where the directions field must be expanded
    (default is the whole image when set to None).
    :type mask: numpy.ndarray

    :return: The filtered image with the adjunct operator.
    :rtype: numpy.ndarray
    """
    guide = np.zeros(image.shape)
    guide[:] = -np.inf

    regularized = np.zeros(directions.shape)

    if mask is None:
        mask = np.ones(image.shape).astype(bool)

    ki = structuring_elements[0, 0].shape[0] // 2
    kj = structuring_elements[0, 0].shape[1] // 2

    oj, oi = np.meshgrid(range(-kj, kj + 1), range(-ki, ki + 1))

    field_expansion = np.zeros((2,) + oi.shape)

    for i in range(ki, image.shape[0] - ki):
        for j in range(kj, image.shape[1] - kj):
            if mask[i, j]:
                expansion = np.add(image[i, j], structuring_elements[i, j])
                condition = np.less(guide[i + oi, j + oj], expansion)

                guide[i + oi[condition], j + oj[condition]] = \
                    expansion[condition]

                field_expansion[0] = np.add(directions[0, i, j],
                                            structuring_elements[i, j])
                field_expansion[1] = np.add(directions[1, i, j],
                                            structuring_elements[i, j])

                regularized[:, i + oi[condition], j + oj[condition]] = \
                    field_expansion[:, condition]

    return regularized

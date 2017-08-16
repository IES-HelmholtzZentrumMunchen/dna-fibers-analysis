"""
Module for grayscale morphology operations with varying structuring elements,
used for reconstruction during the fibers detection.
"""
import numpy as np


def varying_filtering_2d(image, structuring_elements, function_map,
                         function_reduce, mask):
    """Compute a grayscale variant morphological filtering.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    function_map : callable function
        Function of the filter domain used when mapping.

    function_reduce : callable function
        Function of the filter domain used when reducing.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The filtered image.
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
    """Compute a grayscale variant dilation.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The dilated image.

    See Also
    --------
    varying_filtering_2d : Base morphological filtering method
    """
    return varying_filtering_2d(image, structuring_elements,
                                np.add, np.max, mask)


def varying_erosion(image, structuring_elements, mask):
    """Compute a grayscale variant erosion.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The eroded image.

    See Also
    --------
    varying_filtering_2d : Base morphological filtering method
    """
    return varying_filtering_2d(image, structuring_elements,
                                np.subtract, np.min, mask)


def varying_opening(image, structuring_elements, mask):
    """Compute a grayscale variant opening.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The opened image.

    See Also
    --------
    varying_filtering_2d : Base grayscale variant filtering method.
    varying_dilation : Grayscale variant dilation method.
    varying_erosion : Grayscale variant erosion method.
    """
    return varying_dilation(varying_erosion(image, structuring_elements, mask),
                            structuring_elements, mask)


def varying_closing(image, structuring_elements, mask):
    """Compute a grayscale varying closing.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The closed image.

    See Also
    --------
    varying_filtering_2d : Base grayscale variant filtering method.
    varying_dilation : Grayscale variant dilation method.
    varying_erosion : Grayscale variant erosion method.
    """
    return varying_erosion(varying_dilation(image, structuring_elements, mask),
                           structuring_elements, mask)


def adjunct_varying_filtering_2d(image, structuring_elements, function_map,
                                 function_reduce, initialization, mask):
    """Compute a grayscale variant morphological filtering with adjunct operator.

    This is particularly important when using varying structuring elements, as
    the structuring are not, by definition, necessarily symmetric.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    function_map : callable function
        Function of the filter domain used when mapping.

    function_reduce : callable function
        Function of the filter domain used when reducing.

    initialization : float or int
        Initialization element.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The filtered image with the adjunct operator.
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
    """Compute a grayscale variant dilation with adjunct operator.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The dilated image with the adjunct of the erosion.

    See Also
    --------
    adjunct_varying_filtering_2d : Base grayscale variant filtering method with
        adjunct operator.
    """
    return adjunct_varying_filtering_2d(image, structuring_elements,
                                        np.add, np.maximum, -np.inf, mask)


def adjunct_varying_erosion(image, structuring_elements, mask):
    """Compute a grayscale variant erosion with adjunct operator.

    Parameters
    ----------
    image: numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        The operations will be processed only inside the mask.

    Returns
    -------
    numpy.ndarray
        The eroded image with the adjunct of the dilation.

    See Also
    --------
    adjunct_varying_filtering_2d : Base grayscale variant filtering method with
        adjunct operator.
    """
    return adjunct_varying_filtering_2d(image, structuring_elements,
                                        np.subtract, np.minimum, np.inf, mask)


def adjunct_varying_opening(image, structuring_elements, mask=None,
                            extent_mask=None, adjunct_dilation=True):
    """Compute a grayscale variant opening with adjunct operator.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        Mask of where could be object of interest (default is the
        whole image when set to None).

    extent_mask : numpy.ndarray
        Mask of where the structuring elements could reach
        (default is the whole image when set to None). It should contain
        the mask.

    adjunct_dilation : bool
        Use adjunct of erosion as dilation (True by default).

    Returns
    -------
    numpy.ndarray
        The opened image.

    See Also
    --------
    adjunct_varying_filtering_2d : Base grayscale variant filtering method with
        adjunct operator.
    adjunct_varying_dilation : Grayscale variant dilation method with adjunct
        operator.
    adjunct_varying_erosion : Grayscale variant erosion method with adjunct
        operator.
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
    """Compute a grayscale variant closing with adjunct operator.

    Parameters
    ----------
    image : numpy.ndarray
        Image to filter.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        Mask of where could be object of interest (default is the
        whole image when set to None).

    extent_mask : numpy.ndarray
        Mask of where the structuring elements could reach
        (default is the whole image when set to None). It should contain
        the mask.

    adjunct_dilation : bool
        Use adjunct of erosion as dilation (True by default).

    Returns
    -------
    numpy.ndarray
        The closed image.

    See Also
    --------
    adjunct_varying_filtering_2d : Base grayscale variant filtering method with
        adjunct operator.
    adjunct_varying_dilation : Grayscale variant dilation method with adjunct
        operator.
    adjunct_varying_erosion : Grayscale variant erosion method with adjunct
        operator.
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
    """Regularize a vector field using guided morphological dilation.

    It computes grayscale adjunct filtering of the vector field, guided with an
    image and with a varying structuring element.

    Parameters
    ----------
    image : numpy.ndarray
        Image used as guide.

    directions : numpy.ndarray
        The directions to regularize.

    structuring_elements : dict
        Structuring elements possibly varying.

    mask : numpy.ndarray
        Mask to limit where the directions field must be expanded (default is
        the whole image when set to None).

    Returns
    -------
    numpy.ndarray
        The filtered image with the adjunct operator.

    See Also
    --------
    varying_filtering_2d : Base grayscale variant filtering method.
    varying_dilation : Grayscale variant dilation method.
    varying_erosion : Grayscale variant erosion method.
    adjunct_varying_filtering_2d : Base grayscale variant filtering method with
        adjunct operator.
    adjunct_varying_dilation : Grayscale variant dilation method with adjunct
        operator.
    adjunct_varying_erosion : Grayscale variant erosion method with adjunct
        operator.
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

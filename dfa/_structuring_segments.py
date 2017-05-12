"""
Module for structuring segments creation, that can be used for morphological
operations.
"""
import numpy as np
from copy import copy
from scipy.special import erf

import warnings


def bresenham_segment(p1, p2, k):
    """
    Compute a discrete segment between discrete points p1 and p2 using
    Bresenham's algorithm.

    :param p1: First point of the segment.
    :type p1: list of int

    :param p2: Second point of the segment.
    :type p2: list of int

    :param k: Half-size of the image domain (full size is 2*k+1).
    :type k: tuple of int

    :return: A binary image containing the discrete segment.
    :rtype: numpy.ndarray
    """
    def _trace(image, k, p):
        image[p[1] + k[1], p[0] + k[0]] = 1

    def _process_octant(d, i, comp, s, p1, p2, image, k):
        e = d[i]
        d[0] *= 2
        d[1] *= 2

        _trace(image, k, p1)
        p1[i] += s[0]
        while p1[i] != p2[i]:
            e += s[1] * d[1 - i]
            if comp(e, 0):
                p1[1 - i] += s[2]
                e += s[3] * d[i]
            _trace(image, k, p1)
            p1[i] += s[0]

    image = np.zeros((2 * k[1] + 1, 2 * k[0] + 1))

    _p1 = copy(p1)
    dx = p2[0] - _p1[0]
    dy = p2[1] - _p1[1]

    if dx != 0:
        if dx > 0:
            if dy != 0:
                if dy > 0:
                    # first quadrant
                    if dx >= dy:
                        # first octant
                        _process_octant([dx, dy], 0, lambda a, b: a < b,
                                        [1, -1, 1, 1], _p1, p2, image, k)
                    else:
                        # second octant
                        _process_octant([dx, dy], 1, lambda a, b: a < b,
                                        [1, -1, 1, 1], _p1, p2, image, k)
                else:  # dy < 0 (and dx > 0)
                    # fourth quadrant
                    if dx >= -dy:
                        # eigth octant
                        _process_octant([dx, dy], 0, lambda a, b: a < b,
                                        [1, 1, -1, 1], _p1, p2, image, k)
                    else:
                        # seventh octant
                        _process_octant([dx, dy], 1, lambda a, b: a > b,
                                        [-1, 1, 1, 1], _p1, p2, image, k)
            else:  # dy = 0 (and dx > 0)
                _trace(image, k, _p1)
                _p1[0] += 1
                while _p1[0] != p2[0]:
                    _trace(image, k, _p1)
                    _p1[0] += 1
        else:  # dx < 0
            if dy != 0:
                if dy > 0:
                    # 2nd quadrant
                    if -dx >= dy:
                        # fourth octant
                        _process_octant([dx, dy], 0, lambda a, b: a >= b,
                                        [-1, 1, 1, 1], _p1, p2, image, k)
                    else:
                        # third octant
                        _process_octant([dx, dy], 1, lambda a, b: a <= b,
                                        [1, 1, -1, 1], _p1, p2, image, k)
                else:  # dy < 0 (and dx < 0)
                    # third quadrant
                    if dx <= dy:
                        # fifth octant
                        _process_octant([dx, dy], 0, lambda a, b: a >= b,
                                        [-1, -1, -1, 1], _p1, p2, image, k)
                    else:
                        # sixth octant
                        _process_octant([dx, dy], 1, lambda a, b: a >= b,
                                        [-1, -1, -1, 1], _p1, p2, image, k)
            else:  # dy = 0 (and dx < 0)
                _trace(image, k, _p1)
                _p1[0] -= 1
                while _p1[0] != p2[0]:
                    _trace(image, k, _p1)
                    _p1[0] -= 1
    else:  # dx = 0
        if dy != 0:
            if dy > 0:
                _trace(image, k, _p1)
                _p1[1] += 1
                while _p1[1] != p2[1]:
                    _trace(image, k, _p1)
                    _p1[1] += 1
            else:  # dy < 0 (and dx = 0)
                _trace(image, k, _p1)
                _p1[1] -= 1
                while _p1[1] != p2[1]:
                    _trace(image, k, _p1)
                    _p1[1] -= 1

    _trace(image, k, p2)

    return image


def _lineq(p1, p2):
    """
    Return the equation of a line in 2D as a lambda function.

    :param p1: First point of the line.
    :type p1: list of int

    :param p2: Second point of the line.
    :type p2: list of int

    :return: Equation of the line as a lambda function.
    :rtype: lambda function
    """
    a = p1[0] - p2[0]
    b = p2[1] - p1[1]
    c = -(p1[0] * b + p1[1] * a)
    return lambda x, y: a * x + b * y + c


def flat_structuring_segment(direction, thickness, length, k):
    """
    Trace an oriented segment with the given properties using Bresenham's
    algorithm.

    The output is a flat grayscale structuring segment, so the discrete segment
    is denoted by 0 and the background is denoted by -inf.

    :param direction: Orientation vector.
    :type direction: tuple of float

    :param thickness: Thickness of the segment.
    :type thickness: strictly positive float

    :param length: Length of the segment.
    :type length: strictly positive float

    :param k: Half-size of the image domain (full size is 2*k+1).
    :type k: int

    :return: The structuring segment with given properties.
    :rtype: numpy.ndarray
    """
    v_norm = np.sqrt(np.power(direction, 2).sum())

    if v_norm > 0:
        b = length / 2
        direction = np.divide(direction, v_norm)

        if thickness > 1:
            vn = np.array([-direction[1], direction[0]])
            t = thickness // 2
            p1 = np.round(np.multiply(-b, direction) +
                          np.multiply(-t, vn)).astype('int').tolist()
            p2 = np.round(np.multiply(-b, direction) +
                          np.multiply(t, vn)).astype('int').tolist()
            p3 = np.round(np.multiply(b, direction) +
                          np.multiply(t, vn)).astype('int').tolist()
            p4 = np.round(np.multiply(b, direction) +
                          np.multiply(-t, vn)).astype('int').tolist()
            segment = bresenham_segment(p1, p2, (k, k)) + \
                bresenham_segment(p2, p3, (k, k)) + \
                bresenham_segment(p3, p4, (k, k)) + \
                bresenham_segment(p4, p1, (k, k))
            segment[segment > 0] = 1

            x, y = np.meshgrid(range(-k, k + 1), range(-k, k + 1))
            inside = np.bitwise_and(
                _lineq(p1, p2)(y, x) >= 0,
                np.bitwise_and(
                    _lineq(p2, p3)(y, x) >= 0,
                    np.bitwise_and(
                        _lineq(p3, p4)(y, x) >= 0,
                        _lineq(p4, p1)(y, x) >= 0)))
            segment[inside] = 1
        else:
            p1 = np.round(np.multiply(-b, direction)).astype('int').tolist()
            p2 = np.round(np.multiply(b, direction)).astype('int').tolist()
            segment = bresenham_segment(p1, p2, (k, k))
    else:
        segment = np.zeros((2 * k + 1, 2 * k + 1))
        segment[k, k] = 1

    # Set up the values for flat grayscale structuring elements
    segment[segment == 0] = -np.inf
    segment[segment == 1] = 0

    return segment


def bandlimited_structuring_segment(direction, thickness, length, k, scaling):
    """
    Build a discrete and band-limited segment with Gaussian
    functions from given orientation vector, length and thickness.

    Since it produces a non-flat grayscale structuring element, one needs
    to provide a scaling factor. The output image will have intensity
    values in range from -scaling (background) to 0 (segment).

    :param direction: Unitary orientation vector (it will be normalized).
    :type direction: tuple of floats

    :param thickness: Thickness of the segment.
    :type thickness: float or integer

    :param length: Length of the segment.
    :type length: float or integer

    :param k: Half-size of the image domain (full size is 2*k+1).
    :type k: int

    :param scaling: Scaling factor of the segment (cf. description).
    :type scaling: float or integer

    :return: Segment with given orientation, thickness and length.
    :rtype: numpy.ndarray
    """
    va = np.divide(direction, np.sqrt(np.power(direction, 2).sum()))
    vb = [-va[1], va[0]]  # Vector normal to va

    x, y = np.meshgrid(range(-k, k + 1), range(-k, k + 1))

    segment = scaling * (np.exp(-0.5 * np.power((y * va[1] + x * va[0]), 2) /
                                thickness ** 2) * 0.5 *
                         (1 + erf(np.divide(length - 2 * np.sqrt(np.power(
                             x * vb[0] + y * vb[1], 2)), 2 * thickness))) - 1)

    # Set up the values for non-flat grayscale structuring elements
    segment[segment == -scaling] = -np.inf

    return segment


def _angles2vectors(angles):
    """
    Helper function to convert angles in degrees to unit vectors.

    The angles corresponds to the angles between the returned vectors and (1,0).

    :param angles: Angles in degrees.
    :type angles: iterable of float or int

    :return: The corresponding unit vectors.
    :rtype: numpy.ndarray
    """
    return np.column_stack((np.cos(np.multiply(np.pi, angles) / 180),
                            np.sin(np.multiply(np.pi, angles) / 180)))


def _vectors2angles(vectors):
    """
    Helper function to convert vectors in angles in degrees.

    The angles are the angles between the input vectors and (1,0).

    :param vectors: Vector field of size (2, i, j).
    :type vectors: numpy.ndarray

    :return: The corresponding angles with field's shape preserved.
    :rtype: numpy.ndarray
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return 180 * np.arccos(
            vectors[0, :] / np.sqrt(np.power(vectors, 2).sum(axis=0))) / np.pi


def _segments_family(angles, thickness, length, k, scaling=100, flat=True):
    """
    Compute a family of structuring segments from angles.

    By default, it compute flat grayscale structuring elements.

    :param angles: Angles for which to compute the segments.
    :type angles: iterable of float or int

    :param thickness: Thickness of the segments.
    :type thickness: strictly positive float

    :param length: Length of the segments.
    :type length: strictly positive float

    :param k: Half-size of the image domain (full size is 2*k+1).
    :type k: int

    :param scaling: Scaling factor of the segment (for non-flat grayscale
    structuring elements only, default is 100).
    :type scaling: float or integer

    :param flat: The method used to generate (default is True, which uses
    Bresenham's algorithm). When False, it uses a bandlimited segment and needs
    a scaling factor.
    :type flat: bool

    :return: Family of segments.
    :rtype: dict
    """
    vectors = _angles2vectors(angles)
    family = dict()

    if flat:
        for angle, vector in zip(angles, vectors):
            family[angle] = flat_structuring_segment(vector, thickness,
                                                     length, k)
    else:
        for angle, vector in zip(angles, vectors):
            family[angle] = bandlimited_structuring_segment(vector, thickness,
                                                            length, k,
                                                            scaling)

    family[np.array(np.nan).astype('int').tolist()] = \
        flat_structuring_segment((0, 0), thickness, length, k)

    return family


def structuring_segments(directions, thickness, length, scaling=100, flat=True):
    """
    Compute the structuring segments corresponding to the input vector field.

    By default, flat grayscale structuring elements are used.

    :param directions: Vector field of size (2, i, j).
    :type directions: numpy.ndarray

    :param thickness: Thickness of the segments.
    :type thickness: strictly positive float

    :param length: Length of the segments.
    :type length: strictly positive float

    :param scaling: Scaling factor of the segment (for non-flat grayscale
    structuring elements only, default is 100).
    :type scaling: float or integer

    :param flat: The method used to generate (default is True, which uses
    Bresenham's algorithm). When False, it uses a bandlimited segment and needs
    a scaling factor.
    :type flat: bool

    :return: Map of structuring segments from image space to structuring
    elements space.
    :rtype: dict
    """
    k = length // 2 + 1
    family = _segments_family(range(180), thickness, length, k, scaling, flat)
    angles = np.mod(np.round(_vectors2angles(directions)), 180).astype('int')

    segments = dict()
    for i in range(directions.shape[1]):
        for j in range(directions.shape[2]):
            segments[i, j] = family[angles[i, j]]

    return segments

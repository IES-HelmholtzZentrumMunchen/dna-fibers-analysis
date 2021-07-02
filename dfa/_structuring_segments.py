"""
Module for structuring segments creation, that can be used for morphological
operations.
"""
import numpy as np
from copy import copy
from scipy.special import erf

import warnings


def bresenham_segment(p1, p2, k):
    """Compute a discrete segment between discrete points p1 and p2 using
    Bresenham's algorithm.

    Parameters
    ----------
    p1 : List[int]
        First point of the segment.

    p2 : List[int]
        Second point of the segment.

    k : (int,int)
        Half-size of the image domain (full size is 2*k+1).

    Returns
    -------
    numpy.ndarray
        A binary image containing the discrete segment.
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
    """Return the equation of a line in 2D as a lambda function.

    Parameters
    ----------
    p1 : List[int]
        First point of the line.

    p2 : List[int]
        Second point of the line.

    Returns
    -------
    lambda function
        Equation of the line as a lambda function.
    """
    a = p1[0] - p2[0]
    b = p2[1] - p1[1]
    c = -(p1[0] * b + p1[1] * a)
    return lambda x, y: a * x + b * y + c


def flat_structuring_segment(direction, thickness, length, k):
    """Trace an oriented segment with the given properties using Bresenham's
    algorithm.

    The output is a flat grayscale structuring segment, so the discrete segment
    is denoted by 0 and the background is denoted by -inf.

    Parameters
    ----------
    direction : tuple of float
        Orientation vector.

    thickness : strictly positive float
        Thickness of the segment.

    length : strictly positive float
        Length of the segment.

    k : int
        Half-size of the image domain (full size is 2*k+1).

    Returns
    -------
    numpy.ndarray
        The structuring segment with given properties.

    See Also
    --------
    bandlimited_structuring_segment : Generate a grayscale structuring segment.
    structuring_segments : Generate structuring segments from a vector field.
    """
    v_norm = np.sqrt(np.power(direction, 2).sum())

    if v_norm > 0:
        b = length / 2
        direction = np.divide(direction, v_norm)

        if thickness > 1:
            segment = _extracted_from_flat_structuring_segment_10(
                direction, thickness, b, k
            )

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

def _extracted_from_flat_structuring_segment_10(direction, thickness, b, k):
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
    result = bresenham_segment(p1, p2, (k, k)) + \
                bresenham_segment(p2, p3, (k, k)) + \
                bresenham_segment(p3, p4, (k, k)) + \
                bresenham_segment(p4, p1, (k, k))
    result[result > 0] = 1

    x, y = np.meshgrid(range(-k, k + 1), range(-k, k + 1))
    inside = np.bitwise_and(
        _lineq(p1, p2)(y, x) >= 0,
        np.bitwise_and(
            _lineq(p2, p3)(y, x) >= 0,
            np.bitwise_and(
                _lineq(p3, p4)(y, x) >= 0,
                _lineq(p4, p1)(y, x) >= 0)))
    result[inside] = 1
    return result


def bandlimited_structuring_segment(direction, thickness, length, k, scaling):
    """Build a discrete and band-limited segment with Gaussian
    functions from given orientation vector, length and thickness.

    Since it produces a non-flat grayscale structuring element, one needs
    to provide a scaling factor. The output image will have intensity
    values in range from -scaling (background) to 0 (segment).

    Parameters
    ----------
    direction : tuple of floats
        Unitary orientation vector (it will be normalized).

    thickness : float or integer
        Thickness of the segment.

    length : float or integer
        Length of the segment.

    k : int
        Half-size of the image domain (full size is 2*k+1).

    scaling : float or integer
        Scaling factor of the segment (cf. description).

    Returns
    -------
    numpy.ndarray
        Segment with given orientation, thickness and length.

    See Also
    --------
    flat_structuring_segment : Generate a flat grayscale structuring segment.
    structuring_segments : Generate structuring segments from a vector field.
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
    """Helper function to convert angles in degrees to unit vectors.

    The angles corresponds to the angles between the returned vectors and (1,0).

    Parameters
    ----------
    angles : iterable of float or int
        Angles in degrees.

    Returns
    -------
    numpy.ndarray
        The corresponding unit vectors.
    """
    return np.column_stack((np.cos(np.multiply(np.pi, angles) / 180),
                            np.sin(np.multiply(np.pi, angles) / 180)))


def _vectors2angles(vectors):
    """Helper function to convert vectors in angles in degrees.

    The angles are the angles between the input vectors and (1,0).

    Parameters
    ----------
    vectors : numpy.ndarray
        Vector field of size (2, i, j).

    Returns
    -------
    numpy.ndarray
        The corresponding angles with field's shape preserved.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return 180 * np.arccos(
            vectors[0, :] / np.sqrt(np.power(vectors, 2).sum(axis=0))) / np.pi


def _segments_family(angles, thickness, length, k, scaling=100, flat=True):
    """Compute a family of structuring segments from angles.

    By default, it compute flat grayscale structuring elements.

    Parameters
    ----------
    angles : iterable of float or int
        Angles for which to compute the segments.

    thickness : strictly positive float
        Thickness of the segments.

    length : strictly positive float
        Length of the segments.

    k : int
        Half-size of the image domain (full size is 2*k+1).

    scaling : float or integer
        Scaling factor of the segment (for non-flat grayscale
        structuring elements only, default is 100).

    flat : bool
        The method used to generate (default is True, which uses Bresenham's
        algorithm). When False, it uses a bandlimited segment and needs a
        scaling factor.

    Returns
    -------
    dict
        Family of segments.
    """
    vectors = _angles2vectors(angles)
    family = {}

    for angle, vector in zip(angles, vectors):
        if flat:
            family[angle] = flat_structuring_segment(vector, thickness,
                                                     length, k)
        else:
            family[angle] = bandlimited_structuring_segment(vector, thickness,
                                                            length, k,
                                                            scaling)

    family[np.array(np.nan).astype('int').tolist()] = \
        flat_structuring_segment((0, 0), thickness, length, k)

    return family


def structuring_segments(directions, thickness, length, scaling=100, flat=True):
    """Compute the structuring segments corresponding to the input vector field.

    By default, flat grayscale structuring elements are used.

    Parameters
    ----------
    directions : numpy.ndarray
        Vector field of size (2, i, j).

    thickness : strictly positive float
        Thickness of the segments.

    length : strictly positive float
        Length of the segments.

    scaling : float or integer
        Scaling factor of the segment (for non-flat grayscale structuring
        elements only, default is 100).

    flat : bool
        The method used to generate (default is True, which uses Bresenham's
        algorithm). When False, it uses a bandlimited segment and needs a
        scaling factor.

    Returns
    -------
    dict
        Map of structuring segments from image space to structuring elements
        space.

    See Also
    --------
    flat_structuring_segment : Generate a flat grayscale structuring segment.
    bandlimited_structuring_segment : Generate a grayscale structuring segment.
    """
    k = length // 2 + 1
    family = _segments_family(range(180), thickness, length, k, scaling, flat)
    angles = np.mod(np.round(_vectors2angles(directions)), 180).astype('int')

    segments = {}
    for i in range(directions.shape[1]):
        for j in range(directions.shape[2]):
            segments[i, j] = family[angles[i, j]]

    return segments

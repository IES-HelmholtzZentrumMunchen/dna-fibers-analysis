"""
Extraction module of the DNA fiber analysis package.

Use this module to compare fiber paths, extract fiber profiles and images.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline


def _compute_normals(fiber):
    """Compute normals along a fiber path.

    Computing the tangent needs 5 points; therefore 2 points at the
    beginning and 2 points at the end will not be used for the unfolding.
    This issue can be overcome by extrapolation(e.g.linear).
    In our case, it represents only a fiber path shortened by 2 pixels at
    the beginning and 2 pixels at the end. So we omit a correction for
    that issue.

    Parameters
    ----------
    fiber : numpy.ndarray
        Fiber coordinates.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The coordinates along fiber path with their corresponding normals.
    """
    points = []
    normals = []

    for i in range(2, fiber.shape[1] - 2):
        # Instead using finite differences, the tangent vector is computed
        # at point p with a least-square linear fit 5 points in total (the
        # central point 2 points before and 2 points after) in order to
        # avoid numerical issues.
        x1, y1 = fiber[:, i - 2]
        x2, y2 = fiber[:, i - 1]
        x3, y3 = fiber[:, i]
        x4, y4 = fiber[:, i + 1]
        x5, y5 = fiber[:, i + 2]

        slope_num = (x1 * y2 - 4 * x1 * y1 + x2 * y1 + x1 * y3 - 4 * x2 * y2 +
                     x3 * y1 + x1 * y4 + x2 * y3 + x3 * y2 + x4 * y1 + x1 *
                     y5 + x2 * y4 - 4 * x3 * y3 + x4 * y2 + x5 * y1 + x2 * y5 +
                     x3 * y4 + x4 * y3 + x5 * y2 + x3 * y5 - 4 * x4 * y4 + x5 *
                     y3 + x4 * y5 + x5 * y4 - 4 * x5 * y5)
        slope_den = (2 * (- 2 * x1 * x1 + x1 * x2 + x1 * x3 + x1 * x4 + x1 *
                          x5 - 2 * x2 * x2 + x2 * x3 + x2 * x4 + x2 * x5 - 2 *
                          x3 * x3 + x3 * x4 + x3 * x5 - 2 * x4 * x4 + x4 * x5 -
                          2 * x5 * x5))

        # if slope denominator is null, it means horizontal normal
        # (vertical tangent), so set normal to (1, 0)
        if slope_den != 0:
            slope = slope_num / slope_den

            # parametric formula that makes unit vector
            x = np.sqrt(1 / (1 + slope * slope))

            # in 2D, orthogonal vector is unique, so closed - form solution
            normal = [-slope * x, x]
        else:
            normal = [1, 0]

        # Fix heterogeneous orientation of normal vector in order to get
        # consistent results (e.g. image unfolding).
        if i > 2:
            last_normal = normals[-1]

            # Since vectors are unit, if the dot product is negative, they
            # have opposite orientations
            if np.less(normal[0] * last_normal[0] +
                       normal[1] * last_normal[1], 0):
                normal = np.negative(normal)

        points.append([x3, y3])
        normals.append(normal)

    return np.array(points).T, np.array(normals).T


def unfold_fibers(image, fibers, radius=4):
    """Unfold the fibers in image.

    Sampled points of normals along the multiple fibers axis are interpolated
    and returned as an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    fibers : List[numpy.ndarray]
        Input fibers.

    radius : 0 < int
        Radius of the band along fiber axis to extract (default is 4).

    Returns
    -------
    List[numpy.ndarray]
        The unfolded fibers as images.

    See Also
    --------
        extract_fibers : extract fibers in multiple images.
    """
    f = [RectBivariateSpline(range(channel.shape[0]),
                             range(channel.shape[1]),
                             channel)
         for channel in image]

    unfolded_fibers = []

    for fiber in fibers:
        points, normals = _compute_normals(fiber)
        unfolded_fiber = np.zeros((len(f), 2 * radius + 1, points.shape[1]))

        for c in range(len(f)):
            for x in range(points.shape[1]):
                for y in range(2 * radius + 1):
                    s = y - radius
                    unfolded_fiber[c, y, x] = f[c](
                        points[1, x] + s * normals[1, x],
                        points[0, x] + s * normals[0, x])

        unfolded_fibers.append(unfolded_fiber)

    return unfolded_fibers


def extract_fibers(images, fibers, radius=4):
    """Extract the fibers in images.

    Parameters
    ----------
    images : List[numpy.ndarray]
        Input images.

    fibers : List[List[numpy.ndarray]]
        Input fibers.

    radius : 0 < int
        Radius of the band along fiber axis to extract (default is 4).

    Returns
    -------
    List[List[numpy.ndarray]]
        The extracted fibers for each image.

    See Also
    --------
        unfold_fibers : extract fibers in a single image.
    """
    extracted_fibers = []

    for image, image_fibers in zip(images, fibers):
        extracted_fibers.append(unfold_fibers(image, image_fibers, radius))

    return extracted_fibers


def extract_profiles_from_fiber(fiber, func=np.mean):
    """Extract profiles from fiber.

    The fiber is an image of the unfolded fiber path with a given height
    (corresponding to the radius). The profiles are extracted by applying the
    func function to each column of the fiber image (default is mean).

    If the profiles have strange values (e.g. below zero, possibly due to
    interpolation processes), these values are dropped.

    Parameters
    ----------
    fiber : numpy.ndarray
        Input fiber from which to extract profiles.

    func : callable function
        Function used to reduce the columns of the fiber image (default is
        mean).

    Returns
    -------
    numpy.ndarray
        The profiles of the fiber as a column-oriented array (x, y1, y2).
    """
    profiles = np.vstack((range(fiber.shape[2]),
                          func(fiber[0], axis=0),
                          func(fiber[1], axis=0))).T

    profiles = profiles[np.all(np.greater_equal(profiles[:, 1:], 0), axis=1)]
    profiles = profiles[np.all(np.bitwise_not(np.isnan(profiles)), axis=1)]
    profiles = profiles[np.all(np.bitwise_not(np.isinf(profiles)), axis=1)]

    return profiles

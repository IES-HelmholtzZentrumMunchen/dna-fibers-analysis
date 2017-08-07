"""
Extraction module of the DNA fiber analysis package.

Use this module to compare fiber paths, extract fiber profiles and images.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline


def coarse_fibers_spatial_distance(f1, f2):
    """
    Coarse spatial distance between two fibers.

    The coarse distance is computed as the euclidian distance between the
    centers of mass of the considered fibers.

    :param f1: First fiber to compare.
    :type f1: numpy.ndarray

    :param f2: Second fiber to compare.
    :type f2: numpy.ndarray

    :return: The coarse spatial distance between fibers (in spatial units).
    :rtype: float
    """
    cm_f1 = f1.mean(axis=1)
    cm_f2 = f2.mean(axis=1)

    return np.linalg.norm(cm_f1 - cm_f2, ord=2)


def coarse_fibers_orientation_distance(f1, f2):
    """
    Coarse orientation distance between two fibers.

    The global orientations of fibers are computed and compared.

    :param f1: First fiber to compare.
    :type f1: numpy.ndarray

    :param f2: Second fiber to compare.
    :type f2: numpy.ndarray

    :return: The coarse orientation distance between fibers (in degrees).
    :rtype: float
    """
    orient_f1 = f1[:, -1] - f1[:, 0]
    orient_f2 = f2[:, -1] - f2[:, 0]

    angle = np.abs((orient_f1 * orient_f2).sum() /
                   (np.linalg.norm(orient_f1, ord=2) *
                    np.linalg.norm(orient_f2, ord=2)))

    if angle > 1:
        angle = 1

    return 180 * np.arccos(angle) / np.pi


def match_fibers_pairs(l1, l2, max_spatial_distance=50,
                       max_orientation_distance=30):
    """
    Match pairs of fibers from two given lists.

    The coarse distance distance between fibers are computed and the
    generated distance map is traversed by minimal distance first to generate
    the pairs, until no pair can be created.

    The fibers are associated once. This means that if one list is bigger than
    the other, there will be some fibers from the biggest that will have
    no match in the other list.

    Also, the maximal distance parameters allow to not associate fibers that
    are to far away from each other and do not share a similar orientation.

    :param l1: First list of fibers.
    :type l1: list of numpy.ndarray

    :param l2: Second list of fibers.
    :type l2: list of numpy.ndarray

    :param max_spatial_distance: Maximal spatial distance accepted to be
    associated (in spatial units, default is 50).
    :type max_spatial_distance: positive float

    :param max_orientation_distance: Maximal orientation distance accepted to
    be associated (in degrees, default is 30).
    :type max_orientation_distance: float within range [0, 180[

    :return: The matched pairs of fibers.
    :rtype: list of tuples of numpy.ndarray
    """
    # Build distance map
    spatial_dist = np.zeros((len(l1), len(l2)))
    orientation_dist = np.zeros((len(l1), len(l2)))

    for i, f1 in enumerate(l1):
        for j, f2 in enumerate(l2):
            spatial_dist[i, j] = coarse_fibers_spatial_distance(f1, f2)
            orientation_dist[i, j] = coarse_fibers_orientation_distance(f1, f2)

    # Find pairs
    for k in range(min(spatial_dist.shape)):
        i, j = np.unravel_index(spatial_dist.argmin(), spatial_dist.shape)

        if spatial_dist[i, j] <= max_spatial_distance and \
           orientation_dist[i, j] <= max_orientation_distance:
            yield i, j
            spatial_dist[i, :] = spatial_dist.max()
            spatial_dist[:, j] = spatial_dist.max()
        else:
            break


def fibers_spatial_distances(f1, f2):
    """
    Pointwise spatial distance between two fibers.

    The distance returned are the mean of minimal distances between fibers
    in a pointwise manner and the modified Hausdorff distance.

    To make distances symmetric, the maximal values of both ways are taken as
    the final results.

    :param f1: First fiber to compare.
    :type f1: numpy.ndarray

    :param f2: Second fiber to compare.
    :type f2: numpy.ndarray

    :return: The spatial distances between fibers (in spatial units) (mean and
    Hausdorff).
    :rtype: tuple of floats
    """
    def _closest_distances(f1, f2):
        closest_distances = []

        for p in f1.T:
            min_distance = np.linalg.norm(p - f2[:, 0], ord=2)

            for q in f2.T[1:]:
                distance = np.linalg.norm(p - q, ord=2)

                if distance < min_distance:
                    min_distance = distance

            closest_distances.append(min_distance)

        return closest_distances

    closest_distances_f1 = _closest_distances(f1, f2)
    closest_distances_f2 = _closest_distances(f2, f1)

    return (max(np.mean(closest_distances_f1), np.mean(closest_distances_f2)),
            max(np.median(closest_distances_f1),
                np.median(closest_distances_f2)),
            max(np.max(closest_distances_f1), np.max(closest_distances_f2)))


def _compute_normals(fiber):
    """
    Compute normals along a fiber path.

    Computing the tangent needs 5 points; therefore 2 points at the
    beginning and 2 points at the end will not be used for the unfolding.
    This issue can be overcome by extrapolation(e.g.linear).
    In our case, it represents only a fiber path shortened by 2 pixels at
    the beginning and 2 pixels at the end. So we omit a correction for
    that issue.

    :param fiber: Fiber coordinates.
    :type fiber: numpy.ndarray

    :return: The coordinates along fiber path with their corresponding normals.
    :rtype: tuple of numpy.ndarray
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

        slope = (x1 * y2 - 4 * x1 * y1 + x2 * y1 + x1 * y3 - 4 * x2 * y2 +
                 x3 * y1 + x1 * y4 + x2 * y3 + x3 * y2 + x4 * y1 + x1 *
                 y5 + x2 * y4 - 4 * x3 * y3 + x4 * y2 + x5 * y1 + x2 * y5 +
                 x3 * y4 + x4 * y3 + x5 * y2 + x3 * y5 - 4 * x4 * y4 + x5 *
                 y3 + x4 * y5 + x5 * y4 - 4 * x5 * y5) / \
                (2 * (- 2 * x1 * x1 + x1 * x2 + x1 * x3 + x1 * x4 + x1 *
                      x5 - 2 * x2 * x2 + x2 * x3 + x2 * x4 + x2 * x5 - 2 *
                      x3 * x3 + x3 * x4 + x3 * x5 - 2 * x4 * x4 + x4 * x5 -
                      2 * x5 * x5))

        # in NaN case, it means horizontal normal (vertical tangent),
        # so initialize to(1, 0)
        normal = [1, 0]

        if not np.isnan(slope):
            # parametric formula that makes unit vector
            x = np.sqrt(1 / (1 + slope * slope))

            # in 2D, orthogonal vector is unique, so closed - form solution
            normal = [-slope * x, x]

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
    """
    Unfold the fibers in image.

    Sampled points of normals along the multiple fiber axis are interpolated
    and returned as an image.

    :param image: Input image.
    :type image: numpy.ndarray

    :param fibers: Input fiber.
    :type fibers: numpy.ndarray

    :param radius: Radius of the band along fiber axis to extract (default
    is 4).
    :type radius: strictly positive int

    :return: The unfolded fiber as an image.
    :rtype: numpy.ndarray
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
    """
    Extract the fibers in images.

    ..see also: unfold_fibers

    :param images: Input images.
    :type images: list of numpy.ndarray

    :param fibers: Input fibers.
    :type fibers: list of list of numpy.ndarray

    :param radius: Radius of the band along fiber axis to extract (default
    is 4).
    :type radius: strictly positive int

    :return: The extracted fibers for each image.
    :rtype: list of list of numpy.ndarray
    """
    extracted_fibers = []

    for image, image_fibers in zip(images, fibers):
        extracted_fibers.append(unfold_fibers(image, image_fibers, radius))

    return extracted_fibers


if __name__ == '__main__':
    import argparse
    import os
    from skimage import io
    from dfa import _utilities as _ut

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=_ut.check_valid_path,
                        help='Path to single image file or path to folder with '
                             'multiple images.')
    parser.add_argument('fibers', type=_ut.check_valid_directory,
                        help='Path to directory containing fiber files.')
    parser.add_argument('--radius', type=_ut.check_positive_int, default=5,
                        help='Radius of the fibers to extract from images '
                             '(default is 5).')
    parser.add_argument('--group-fibers', action='store_true',
                        help='Group the extracted fibers by image.')
    parser.add_argument('--output', type=_ut.check_valid_path,
                        default=None,
                        help='Output path for saving profiles and extracted '
                             'fibers (default is None).')
    args = parser.parse_args()

    # list image files and path
    if os.path.isdir(args.input):
        image_path = args.input
        image_files = os.listdir(image_path)
    else:
        image_path, image_files = os.path.split(args.input)
        image_files = [image_files]

    # read all together
    input_images = []
    input_fibers = []
    input_names = []

    for filename in image_files:
        basename, ext = os.path.splitext(filename)

        if ext == '.tif':
            input_images.append(
                io.imread(os.path.join(image_path, filename)))
            input_fibers.append(
                _ut.read_points_from_txt(args.fibers, basename))
            input_names.append(basename)

    # process
    extracted_fibers = extract_fibers(
        input_images, input_fibers, radius=args.radius)

    # output
    group_axes = None

    if args.group_fibers:
        from matplotlib import pyplot as plt

        group_axes = []

        for image_extracted_fiber, input_name \
                in zip(extracted_fibers, input_names):
            height = 2 * args.radius + 1
            space = 5
            offset = 15
            group_image = np.zeros((
                len(image_extracted_fiber) * height +
                (len(image_extracted_fiber) - 1) * space,
                max(extracted_fiber.shape[2]
                    for extracted_fiber in image_extracted_fiber) + offset,
                3), dtype='uint8')

            for number, extracted_fiber in enumerate(image_extracted_fiber):
                group_image[number * (height + space):
                            number * (height + space) + height,
                            offset:extracted_fiber.shape[2] + offset,
                            0] = 255 * \
                    _ut.norm_min_max(extracted_fiber[0], extracted_fiber)
                group_image[number * (height + space):
                            number * (height + space) + height,
                            offset:extracted_fiber.shape[2] + offset,
                            1] = 255 * \
                    _ut.norm_min_max(extracted_fiber[1], extracted_fiber)

            _, ax = plt.subplots(1, 1)
            ax.imshow(group_image, aspect='equal')

            for number in range(len(image_extracted_fiber)):
                ax.text(0, number * (height + space) + height / 2 + 2,
                        '#{}'.format(number), color='white')

            ax.set_title(input_name)
            ax.axis('off')

            group_axes.append(ax)

    if args.output is None:
        from matplotlib import pyplot as plt

        if group_axes is not None:
            for group_axe in group_axes:
                plt.show(group_axe)
        else:
            for image_extracted_fiber, input_name \
                    in zip(extracted_fibers, input_names):
                for number, extracted_fiber in enumerate(image_extracted_fiber):
                    display_image = np.zeros(extracted_fiber.shape[1:] + (3,),
                                             dtype='uint8')
                    display_image[:, :, 0] = 255 * \
                        _ut.norm_min_max(extracted_fiber[0], extracted_fiber)
                    display_image[:, :, 1] = 255 * \
                        _ut.norm_min_max(extracted_fiber[1], extracted_fiber)

                    _, axes = plt.subplots(nrows=2, ncols=1, sharex='all')

                    axes[0].imshow(display_image, aspect='equal')
                    axes[0].set_title('Unfolded fiber')
                    axes[0].axis('off')

                    axes[1].plot(extracted_fiber[0].sum(axis=0), '-r')
                    axes[1].plot(extracted_fiber[1].sum(axis=0), '-g')
                    axes[1].set_title('Profiles')
                    axes[0].set_xlim(0, extracted_fiber.shape[2])

                    plt.suptitle('{} - fiber #{}'.format(
                        input_name, number + 1))
                    plt.show()
    else:
        raise NotImplementedError

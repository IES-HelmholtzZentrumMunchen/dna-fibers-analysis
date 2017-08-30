"""
Module of utility functions.
"""
import argparse
import os
import shutil
import numpy as np
from scipy.interpolate import splprep, splev
from skimage import io
from matplotlib import pyplot as plt
from matplotlib import collections as coll
import zipfile

from debtcollector import removals
import warnings
warnings.simplefilter('always', category=DeprecationWarning)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@removals.remove
def write_points_to_txt(path, prefix, coordinates):
    """Write points coordinates to text file.

    Parameters
    ----------
    path : str
        Path for output files of fibers.

    prefix : str
        Prefix of the filenames.

    coordinates : list of numpy.ndarray
        Coordinates of the medial axis lines of corresponding fibers.
    """
    for index, points in enumerate(coordinates):
        np.savetxt(
            os.path.join(path, '{}_fiber-{}.txt'.format(prefix, index + 1)),
            points[::-1].T)


@removals.remove
def read_points_from_txt(path, prefix):
    """Read points coordinates from text file.

    Parameters
    ----------
    path : str
        Path to the folder containing the fibers files.

    prefix : str
        Prefix of the fibers filenames.

    Returns
    -------
    list of numpy.ndarray
        Coordinates of the median-axis lines corresponding to fibers and that
        have been red.
    """
    indices = [int(str(os.path.splitext(filename)[0]
                       .split('_')[-1]).split('-')[-1])
               for filename in os.listdir(path)
               if filename.startswith(prefix)]

    return [np.loadtxt(os.path.join(path, '{}_fiber-{}.txt').format(
        prefix, index)).T[::-1] for index in indices]

fiber_indicator = '_fiber-'


def write_fiber(fiber, path, image_name, index):
    """
    Write a single fiber to a text file.

    The parameters are used to format the output filename format. It is defined
    as the following: ::

        <path>/<image_name>_fiber-<index>.txt

    Parameters
    ----------
    fiber : numpy.ndarray
        Points coordinates of the median-axis of the fiber to write. The fiber
        shape must be (2, N) in the order y-x (or i-j).

    path : str
        Path to the directory in which the output file will be written.

    image_name : str
        Name of the image from which the fiber comes from.

    index : 0 < int
        Number of the fiber in that particular image.
    """
    np.savetxt(
        os.path.join(path, '{}{}{}.txt'.format(image_name, fiber_indicator,
                                               index)),
        fiber[::-1].T)


def write_fibers(fibers, path, image_name, indices=None, zipped=False):
    """
    Write multiple fibers into text files.

    The files will be either grouped in a directory or in a zip file, depending
    on the path given. For the output filenames format, please refer to
    write_fiber_from_txt.

    Parameters
    ----------
    fibers : List[numpy.ndarray]
        List of points coordinates of the median-axis of the fibers to write.
        The fibers shapes must be (2, N) in the order y-x (or i-j).

    path : str
        Path to the directory in which the output file will be written.

    image_name : str
        Name of the image from which the fiber comes from.

    indices : None | List[int]
        List of index matching the fibers, or auto-generated index list if
        None is passed (default).

    zipped : bool
        If True, the fibers files will be zipped into the file
        <path>/<image_name>.zip instead of being written in directory
        <path> with filenames <image_name>_fiber-N.txt (default False).
    """
    def _write_fibers(indices, fibers, path, image_name):
        for index, fiber in zip(indices, fibers):
            write_fiber(fiber, path, image_name, index)

    if indices is None:
        indices = range(1, len(fibers) + 1)

    if zipped:
        abs_path = os.path.abspath(path)
        tmp_path = os.path.join(abs_path, '_'.join(['tmp', image_name]))
        os.mkdir(tmp_path)

        cur_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            _write_fibers(indices, fibers, tmp_path, image_name)

            with zipfile.ZipFile(
                    os.path.join(abs_path, '.'.join([image_name, 'zip'])),
                    mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
                for filename in os.listdir('.'):
                    archive.write(filename)
        finally:
            os.chdir(cur_dir)
            shutil.rmtree(tmp_path)

    else:
        _write_fibers(indices, fibers, path, image_name)


def _read_fibers(path):
    """
    Read fiber(s) from file(s).

    The file can be a txt file, a ImageJ roi file or a zip file. For the fiber
    filenames format, please refer to write_fiber_to_txt.

    Parameters
    ----------
    path : str
        Path to the fiber file to be red.

    Returns
    -------
    List[(numpy.ndarray, str, 0 < int)]
        Points coordinates of the median-axis of the fiber, name of the image
        from which the fiber comes from (extracted from filename) and number of
        the fiber in that particular image (also extracted from filename).
    """
    ext = os.path.splitext(path)[-1]

    if ext not in available_readers.keys():
        raise NotImplementedError('There is not reader for {} files '
                                  'implemented yet!'.format(ext))
    return available_readers[ext](path)


def _read_fiber_from_txt(path):
    """
    Read a single fiber from text file.

    For the fiber filenames format, please refer to write_fiber_to_txt.

    Parameters
    ----------
    path : str
        Path to the fiber file to be red.

    Returns
    -------
    List[(numpy.ndarray, str, 0 < int)]
        Points coordinates of the median-axis of the fiber, name of the image
        from which the fiber comes from (extracted from filename) and number of
        the fiber in that particular image (also extracted from filename).
    """
    _, filename = os.path.split(os.path.splitext(path)[0])
    image_name, index = tuple(filename.split(fiber_indicator))
    fiber = np.loadtxt(path).T[::-1]

    return [(fiber, image_name, int(index))]


def _read_fibers_from_zip(path):
    """
    Read fiber(s) from a zip file.

    Each file in the zip file will be red according to its extension.

    Parameters
    ----------
    path : str
        Path to the fiber file to be red.

    Returns
    -------
    List[(numpy.ndarray, str, 0 < int)]
        Points coordinates of the median-axis of the fiber, name of the image
        from which the fiber comes from (extracted from filename) and number of
        the fiber in that particular image (also extracted from filename).
    """
    fibers = []

    tmp_path, zipfilename = os.path.split(os.path.abspath(path))
    basename, ext = tuple(os.path.splitext(zipfilename))

    if ext == '.zip':
        tmp_path = os.path.join(tmp_path, '_'.join(['tmp', basename]))
        os.mkdir(tmp_path)

        try:
            with zipfile.ZipFile(
                    path, mode='r',
                    compression=zipfile.ZIP_DEFLATED) as archive:
                for filename in archive.namelist():
                    archive.extract(filename, path=tmp_path)
                    fibers += _read_fibers(os.path.join(tmp_path, filename))
        finally:
            shutil.rmtree(tmp_path)

    return fibers


def read_fibers(path, image_name=None):
    """
    Read multiple fibers from path or zip file using specified method.

    For the fiber filenames format, please refer to write_fiber_to_txt.

    Parameters
    ----------
    path : str
        Path to the directory or the zip file containing the fiber files.

    image_name : None | str
        If not None, only fibers belonging to image_name will be red (default
        is None).

    Returns
    -------
    List[(numpy.ndarray, str, 0 < int)]
        List containing tuples of points coordinates of the median-axis of the
        fiber, name of the image from which the fiber comes from (extracted
        from filename) and number of the fiber in that particular image (also
        extracted from filename).
    """
    fibers = []

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.find(fiber_indicator) != -1 or \
                            os.path.splitext(filename)[-1] == '.zip':
                fibers += _read_fibers(os.path.join(path, filename))
    else:
        fibers += _read_fibers(path)

    if image_name is not None:
        filtered_fibers = [fiber for fiber in fibers if fiber[1] == image_name]
    else:
        filtered_fibers = fibers

    return filtered_fibers


class ImageJRoiType:
    polygon = 0
    rectangle = 1
    oval = 2
    line = 3
    free_line = 4
    polyline = 5
    no_roi = 6
    freehand = 7
    traced = 8
    angle = 9
    point = 10


class ImageJRoiSizes:
    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256


# noinspection PyTypeChecker
def _read_points_from_imagej_roi(binaries):
    """Read points coordinates from binaries of ImageJ rois.

    This code is based on the ijroi package (https://github.com/tdsmith/ijroi),
    which is based on the gist https://gist.github.com/luispedro/3437255, which
    is finally based on the official ImageJ sources (see http://rsbweb.nih.gov/-
    ij/developer/source/ij/io/RoiDecoder.java.html and http://rsbweb.nih.gov/-
    ij/developer/source/ij/io/RoiEncoder.java.html).

    Parameters
    ----------
    binaries : BufferReader
        Binary data from which to read the points coordinates.

    Returns
    -------
    numpy.ndarray
        Array containing the coordinates in the (I, J) and (Y, X) order.
    """
    def get8():
        word = binaries.read(1)
        if not word:
            raise IOError('Reading ImageJ roi: unexpected EOF!')
        return ord(word)

    def get16():
        word0 = get8()
        word1 = get8()
        return (word0 << 8) | word1

    def get32():
        word0 = get16()
        word1 = get16()
        return (word0 << 16) | word1

    def getfloat():
        value = np.int32(get32())
        return value.view(np.float32)

    magic_number = binaries.read(4)
    if magic_number != b'Iout':
        raise ValueError('Reading ImageJ roi: Magic number not found!')
    _ = get16()  # version

    # It seems that the roi type field occupies 2 Bytes, but only one is used
    roi_type = get8()
    # Discard second Byte:
    get8()
    if roi_type is not ImageJRoiType.polyline and \
       roi_type is not ImageJRoiType.line:
        raise NotImplementedError('Reading ImageJ roi: ROI type {} not '
                                  'supported!'.format(roi_type))

    top = get16()
    left = get16()
    _ = get16()  # bottom
    _ = get16()  # right
    n_coordinates = get16()
    _ = getfloat()  # x1
    _ = getfloat()  # y1
    _ = getfloat()  # x2
    _ = getfloat()  # y2
    _ = get16()  # stroke_width
    _ = get32()  # shape_roi_size
    _ = get32()  # stroke_color
    _ = get32()  # fill_color

    subtype = get16()
    if subtype != 0:
        raise NotImplementedError('Reading ImageJ roi: ROI subtype {} not '
                                  'supported!'.format_map(subtype))
    options = get16()
    _ = get8()  # arrow_style
    _ = get8()  # arrow_head_size
    _ = get16()  # rect_arc_size
    _ = get32()  # position
    _ = get32()  # header2offset

    if options & ImageJRoiSizes.SUB_PIXEL_RESOLUTION:
        get_c = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
        binaries.seek(4 * n_coordinates, 1)
    else:
        get_c = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)

    points[:, 1] = [get_c() for _ in range(n_coordinates)]
    points[:, 0] = [get_c() for _ in range(n_coordinates)]

    if options & ImageJRoiSizes.SUB_PIXEL_RESOLUTION == 0:
        points[:, 1] += left
        points[:, 0] += top

    return points


def _read_fiber_from_imagej_roi(path):
    """
    Read a single fiber from ImageJ ROI file.

    For the fiber filenames format, please refer to write_fiber_to_txt.

    Parameters
    ----------
    path : str
        Path to the fiber file to be red.

    Returns
    -------
    List[(numpy.ndarray, str, 0 < int)]
        Points coordinates of the median-axis of the fiber, name of the image
        from which the fiber comes from (extracted from filename) and number of
        the fiber in that particular image (also extracted from filename).
    """
    _, filename = os.path.split(os.path.splitext(path)[0])
    image_name, index = tuple(filename.split(fiber_indicator))

    with open(path, 'rb') as file:
        return [(_read_points_from_imagej_roi(file).T[::-1],
                 image_name, int(index))]


available_readers = {
    '.txt': _read_fiber_from_txt,
    '.roi': _read_fiber_from_imagej_roi,
    '.zip': _read_fibers_from_zip}


@removals.remove
def read_points_from_imagej_zip(filename):
    """Read points coordinates from zip file containing ImageJ rois.

    Parameters
    ----------
    filename : str
        Path of file to read. It must be a zip file.

    Returns
    -------
    list of numpy.ndarray
        Arrays containing the coordinates in the (I, J) and (Y, X) order.
    """
    with zipfile.ZipFile(filename) as archive:
        return [(n, _read_points_from_imagej_roi(archive.open(n)))
                for n in archive.namelist()]


# noinspection PyUnusedLocal
def _write_points_to_imagej_roi(binaries, coordinates):
    """Write points coordinates to binaries of ImageJ rois.

    This code is based on the ijroi package (https://github.com/tdsmith/ijroi),
    which is based on the gist https://gist.github.com/luispedro/3437255, which
    is finally based on the official ImageJ sources (see http://rsbweb.nih.gov/-
    ij/developer/source/ij/io/RoiDecoder.java.html and http://rsbweb.nih.gov/-
    ij/developer/source/ij/io/RoiEncoder.java.html).

    Parameters
    ----------
    binaries : BufferReader
        Binary data to which to read the points coordinates.

    coordinates : numpy.ndarray
        Coordinates of the points to write to the binaries in the (I, J) or
        (Y, X) order.

    Returns
    -------
    numpy.ndarray
        Array containing the coordinates in the (I, J) and (Y, X) order.
    """
    raise NotImplementedError
#     def put8(word):
#         binaries.write(word)
#
#     def put16(word):
#         put8(word >> 8)
#         put8(word)
#
#     def put32(word):
#         put16(word >> 16)
#         put16(word)
#
#     def putfloat(value):
#         put32(value.view(np.int32))
#
#     binaries.write(b'Iout')  # magic_number
#     put16(np.int16(0))  # version
#
#     # It seems that the roi type field occupies 2 Bytes, but only one is used
#     put8(ImageJRoiType.polyline)
#     put8(np.int8(0))
#
#     put16(np.int16(coordinates[:, 0].min()))  # top
#     put16(np.int16(coordinates[:, 1].min()))  # left
#     put16(np.int16(coordinates[:, 0].max()))  # bottom
#     put16(np.int16(coordinates[:, 1].max()))  # right
#     put16(np.int16(coordinates.shape[0]))  # n_coordinates
#     putfloat(np.float32(0))  # x1
#     putfloat(np.float32(0))  # y1
#     putfloat(np.float32(0))  # x2
#     putfloat(np.float32(0))  # y2
#     put16(np.int16(1))  # stroke_width
#     put32(np.int32(1))  # shape_roi_size
#     put32(np.int32(0))  # stroke_color
#     put32(np.int32(0))  # fill_color
#
#     put16(np.int16(0))  # subtype
#
#     options = put16()
#     _ = put8()  # arrow_style
#     _ = put8()  # arrow_head_size
#     _ = put16()  # rect_arc_size
#     _ = put32()  # position
#     _ = put32()  # header2offset
#
#     if options & ImageJRoiSizes.SUB_PIXEL_RESOLUTION:
#         get_c = putfloat
#         points = np.empty((n_coordinates, 2), dtype=np.float32)
#         binaries.seek(4 * n_coordinates, 1)
#     else:
#         get_c = put16
#         points = np.empty((n_coordinates, 2), dtype=np.int16)
#
#     points[:, 1] = [get_c() for _ in range(n_coordinates)]
#     points[:, 0] = [get_c() for _ in range(n_coordinates)]
#
#     if options & ImageJRoiSizes.SUB_PIXEL_RESOLUTION == 0:
#         points[:, 1] += left
#         points[:, 0] += top
#
#     return points


def check_valid_path(path):
    """ Check for existing path (directory or file). """
    if not os.path.isdir(path) and not os.path.isfile(path):
        raise argparse.ArgumentTypeError('The given path is not a '
                                         'valid path!')

    return path


def check_valid_or_empty_path(path):
    """ Check for existing path (directory or file). """
    if path != '' and not os.path.isdir(path) and not os.path.isfile(path):
        raise argparse.ArgumentTypeError('The given path is not a '
                                         'valid path!')

    return path


def check_valid_directory(path):
    """ Check for existing directory. """
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError('The given path is not a valid '
                                         'directory!')

    return path


def check_valid_file(path):
    """ Check for existing file. """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError('The given path is not a valid file!')

    return path


def check_valid_output_file(path):
    """ Check for existing path where file will be saved. """
    if not os.path.exists(os.path.dirname(path)):
        raise argparse.ArgumentTypeError('The path of the requested output file'
                                         ' is not valid!')

    return path


def check_float_0_1(variable):
    """ Check for floats in ]0, 1]. """
    try:
        variable = float(variable)
    except ValueError:
        raise argparse.ArgumentTypeError('The given variable cannot be '
                                         'converted to float!')

    if variable < 1e-10 or variable > 1:
        raise argparse.ArgumentTypeError('The given variable is out of the '
                                         'valid range (range is ]0, 1])')

    return variable


def check_positive_int(variable):
    """ Check for positive integers. """
    try:
        variable = int(variable)
    except ValueError:
        raise argparse.ArgumentTypeError('The given variable cannot be '
                                         'converted to int!')

    if variable <= 0:
        raise argparse.ArgumentTypeError('The given variable is out of '
                                         'the valid range (range is '
                                         ']0, +inf[).')

    return variable


def check_positive_float(variable):
    """ Check for positive floating point numbers. """
    try:
        variable = float(variable)
    except ValueError:
        raise argparse.ArgumentTypeError('The given variable cannot be '
                                         'converted to float!')

    if variable <= 0:
        raise argparse.ArgumentTypeError('The given variable is out of '
                                         'the valid range (range is '
                                         ']0, +inf[).')

    return variable


@static_vars(n=0, l=[])
def check_scales(variable):
    """ Check the scales validity. """
    try:
        variable = int(variable)
    except ValueError:
        raise argparse.ArgumentTypeError('The given variable cannot be '
                                         'converted to int!')

    if variable <= 0:
        raise argparse.ArgumentTypeError('The given variable is out of '
                                         'the valid range (range is '
                                         ']0, +inf[).')

    if check_scales.n == 1 and variable < check_scales.l[-1]:
        raise argparse.ArgumentTypeError('The second scale must be greater '
                                         'than the first one!')

    check_scales.n += 1
    check_scales.l.append(variable)

    return variable


def norm_min_max(data, norm_data=None):
    """Do a min-max normalization.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.

    norm_data : numpy.ndarray
        Data used to normalize. When set to None, the input data is
        used (default).
    """
    return (data - norm_data.min()) / (norm_data.max() - norm_data.min())


def read_inputs(input_path, mask_path, ext):
    def _read_images_from_path(input_path, ext):
        if os.path.isdir(input_path):
            paths = [os.path.join(input_path, filename)
                     for filename in os.listdir(input_path)
                     if filename.endswith(ext)]
        else:
            paths = [input_path]

        names = [os.path.basename(os.path.splitext(path)[0]) for path in paths]

        # NOTE we would need to use bioformat library to load image data.
        # It is important since the standard skimage reading method does not
        # really support multi-channels composite images.
        return [io.imread(path) for path in paths], names

    images, names = _read_images_from_path(input_path, ext)

    if mask_path == '':
        masks = [None] * len(images)
    else:
        masks, _ = _read_images_from_path(mask_path, ext)

    if len(masks) != len(images):
        raise ValueError('The number of masks is not the same as the'
                         'number of images!')

    return images, names, masks


def create_figures_from_fibers_images(names, extracted_fibers,
                                      radius, group_fibers=False, indices=None,
                                      analysis=None):
    def _channel_to_color(channel):
        if channel == 'IdU':
            return 'green'
        elif channel == 'CIdU':
            return 'red'
        else:
            return 'black'

    if indices is None:
        indices = []
        for image_extracted_fiber in extracted_fibers:
            indices.append(range(1, len(image_extracted_fiber) + 1))

    figures = []

    if group_fibers:
        for image_extracted_fiber, name, fiber_indices \
                in zip(extracted_fibers, names, indices):
            height = 2 * radius + 1
            space = 5
            offset = 15
            group_image = np.zeros((
                len(image_extracted_fiber) * height +
                (len(image_extracted_fiber) - 1) * space,
                max(extracted_fiber.shape[2]
                    for extracted_fiber in image_extracted_fiber) + 2 * offset,
                3), dtype='uint8')

            for number, extracted_fiber in enumerate(image_extracted_fiber):
                group_image[number * (height + space):
                            number * (height + space) + height,
                            offset:extracted_fiber.shape[2] + offset,
                            0] = 255 * \
                    norm_min_max(extracted_fiber[0], extracted_fiber)
                group_image[number * (height + space):
                            number * (height + space) + height,
                            offset:extracted_fiber.shape[2] + offset,
                            1] = 255 * \
                    norm_min_max(extracted_fiber[1], extracted_fiber)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(group_image, aspect='equal')

            # for number in range(len(image_extracted_fiber)):
            for number, index in enumerate(fiber_indices):
                ax.text(0, number * (height + space) + height / 2 + 2,
                        '#{}'.format(index), color='white')

            ax.set_title(name)
            ax.axis('off')

            figures.append(('{}_fibers.png'.format(name), fig))
    else:
        for image_extracted_fiber, name, fiber_indices \
                in zip(extracted_fibers, names, indices):
            for number, extracted_fiber in zip(fiber_indices,
                                               image_extracted_fiber):
                display_image = np.zeros(extracted_fiber.shape[1:] + (3,),
                                         dtype='uint8')
                display_image[:, :, 0] = 255 * \
                    norm_min_max(extracted_fiber[0], extracted_fiber)
                display_image[:, :, 1] = 255 * \
                    norm_min_max(extracted_fiber[1], extracted_fiber)

                x = np.arange(extracted_fiber.shape[2])
                y1 = extracted_fiber[0].sum(axis=0)
                y2 = extracted_fiber[1].sum(axis=0)

                d = analysis.loc[(slice(None), slice(None), number), :]
                channels = d['channel'].tolist()
                pattern = d['pattern'].tolist()[0]
                landmarks = np.insert(d['length'].tolist(),
                                      0, [0]).astype('int').cumsum()
                landmarks[-1] += 1

                regions = [coll.BrokenBarHCollection.span_where(
                    x, ymin=0, ymax=max(y1.max(), y2.max()) + 1,
                    where=np.bitwise_and(x >= landmarks[i],
                                         x <= landmarks[i+1]),
                    facecolor=_channel_to_color(c), alpha=0.25)
                    for i, c in enumerate(channels)]

                fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all')

                axes[0].imshow(display_image, aspect='equal')
                axes[0].set_title('Unfolded fiber')
                axes[0].axis('off')
                axes[0].set_xlim(x.min(), x.max())

                axes[1].plot(extracted_fiber[0].sum(axis=0), '-r')
                axes[1].plot(extracted_fiber[1].sum(axis=0), '-g')
                axes[1].set_title('Profiles ({})'.format(pattern))
                for region in regions:
                    axes[1].add_collection(region)
                axes[1].set_ylim(0, max(y1.max(), y2.max()) + 1)

                fig.suptitle('{} - fiber #{}'.format(name, number))

                figures.append(('{}_fiber-{}.png'.format(name, number), fig))

    return figures


def write_profiles(path, prefix, profiles, indices=None):
    """Write the given set of profiles to the specified path with given prefix.

    Parameters
    ----------
    path : str
        Path where to write.

    prefix : str
        Prefix of the output files.

    profiles : List[numpy.ndarray]
        Set of profiles to write.

    indices : None | List[int]
        List of index matching the fibers, or auto-generated index list if
        None is passed (default).
    """
    if indices is None:
        indices = range(1, len(profiles) + 1)

    for index, profile in zip(indices, profiles):
        np.savetxt(os.path.join(
            path, '{}_fiber-{}.csv'.format(prefix, index)),
            profile,
            delimiter=',', header='X, Y1, Y2', comments='')


def resample_fiber(fiber, rate=2.0):
    """
    Resample a fiber at given rate.

    This method is useful for point-wise comparison of fibers.

    Parameters
    ----------
    fiber : numpy.ndarray
        Input fiber points coordinates to resample.

    rate : 0 < float
        Resampling rate.

    Returns
    -------
    numpy.ndarray
        Resampled fiber points coordinates.
    """
    length = np.sqrt(np.power(np.diff(fiber, axis=1), 2).sum(axis=0)).sum()
    # noinspection PyTupleAssignmentBalance
    coeffs, _ = splprep(fiber, u=np.linspace(0, 1, fiber.shape[1]), s=0, k=1)
    return np.vstack(
        splev(np.linspace(0, 1, np.round(length / rate).astype(int)), coeffs))


def resample_fibers(fibers, rate=2.0):
    """
    Resample fibers at given rate.

    Parameters
    ----------
    fibers : List[numpy.ndarray]
        Input fibers points coordinates to resample.

    rate : 0 < float
        Resampling rate.

    Returns
    -------
    List[numpy.ndarray]
        Resampled fibers points coordinates.
    """
    return [resample_fiber(fiber, rate) for fiber in fibers]

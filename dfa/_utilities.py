"""
Module of utility functions.
"""
import argparse
import os
import numpy as np


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def write_points_to_txt(path, prefix, coordinates):
    """
    Write points coordinates to text file.

    :param path: Path for output files of fibers.
    :type path: str

    :param prefix: Prefix of the filenames.
    :type prefix: str

    :param coordinates: Coordinates of the medial axis lines of corresponding
    fibers.
    :type coordinates: list of numpy.ndarray
    """
    for index, points in enumerate(coordinates):
        np.savetxt(os.path.join(path, '{}_fiber-{}.txt'.format(prefix, index)),
                   points[::-1].T)


def read_points_from_txt(path, prefix):
    """
    Read points coordinates from text file.

    :param path: Path to the folder containing the fibers files.
    :type path: str

    :param prefix: Prefix of the fibers filenames.
    :type prefix: str

    :return: Coordinates of the median-axis lines corresponding to fibers and
    that have been red.
    :rtype: list of numpy.ndarray
    """
    indices = [str(os.path.splitext(filename)[0].split('_')[-1]).split('-')[-1]
               for filename in os.listdir(path)
               if filename.startswith(prefix)]

    return [np.loadtxt(os.path.join(path, '{}_fiber-{}.txt').format(
        prefix, index)).T[::-1] for index in indices]


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


def _read_points_from_imagej_roi(binaries):
    """
    Read points coordinates from binaries of ImageJ rois.

    This code is based on the ijroi package (https://github.com/tdsmith/ijroi),
    which is based on the gist https://gist.github.com/luispedro/3437255, which
    is finally based on the official ImageJ sources (see http://rsbweb.nih.gov/-
    ij/developer/source/ij/io/RoiDecoder.java.html and http://rsbweb.nih.gov/-
    ij/developer/source/ij/io/RoiEncoder.java.html).

    :param binaries: Binary data from which to read the points coordinates.
    :type binaries: _io.BufferReader

    :return: Array containing the coordinates in the (I, J) and (Y, X) order.
    :rtype: numpy.ndarray
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


def read_points_from_imagej_roi(filename):
    """
    Read points coordinates from roi file containing one ImageJ roi.

    :param filename: Path of file to read. It must be a roi file.
    :type filename: str

    :return: Arrays containing the coordinates in the (I, J) and (Y, X) order.
    :rtype: list of numpy.ndarray
    """
    with open(filename, 'rb') as file:
        return _read_points_from_imagej_roi(file)


def read_points_from_imagej_zip(filename):
    """
    Read points coordinates from zip file containing ImageJ rois.

    :param filename: Path of file to read. It must be a zip file.
    :type filename: str

    :return: Arrays containing the coordinates in the (I, J) and (Y, X) order.
    :rtype: list of numpy.ndarray
    """
    import zipfile
    with zipfile.ZipFile(filename) as zipfile:
        return [(n, _read_points_from_imagej_roi(zipfile.open(n)))
                for n in zipfile.namelist()]


# def _write_points_from_imagej_roi(binaries, coordinates):
#     """
#     Write points coordinates to binaries of ImageJ rois.
#
#     This code is based on the ijroi package (https://github.com/tdsmith/ijroi),
#     which is based on the gist https://gist.github.com/luispedro/3437255, which
#     is finally based on the official ImageJ sources (see http://rsbweb.nih.gov/-
#     ij/developer/source/ij/io/RoiDecoder.java.html and http://rsbweb.nih.gov/-
#     ij/developer/source/ij/io/RoiEncoder.java.html).
#
#     :param binaries: Binary data to which to read the points coordinates.
#     :type binaries: _io.BufferReader
#
#     :param coordinates: Coordinates of the points to write to the binaries in
#     the (I, J) or (Y, X) order.
#     :type coordinates: numpy.ndarray
#
#     :return: Array containing the coordinates in the (I, J) and (Y, X) order.
#     :rtype: numpy.ndarray
#     """
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

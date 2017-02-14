"""
Simulation module of the DNA fiber analysis package.

Use this module to simulate synthetic images of DNA fibers. Both fiber
segments and image degradations can be controlled.
"""
import numpy as np
import scipy as sc
from scipy import signal

import skimage as ski
from skimage import transform
from skimage import io

from dfa import model


def fiber(theta, rho, imshape, pattern, length, thickness=1.0, shift=0):
    """
    Simulate a straight fiber image (with no acquisition deterioration).

    The fibers are simulated as straights lines with the Hesse normal form
    parametrization. The arguments of the function make possible to create all
    kinds of straight lines (fibers) by choosing thickness, length and
    shift in the direction of the fiber.

    :param theta: Angle of the Hesse normal form parametrization.
    :type theta: float between -pi/2 and pi/2

    :param rho: Distance to origin of the Hesse normal form parametrization.
    :type rho: float or int

    :param imshape: Shape of the generated image of fiber (2D image only).
    :type imshape: tuple of int with 2 elements

    :param pattern: Channel pattern of the simulated fiber.
    :type pattern: list of int with same size as length

    :param length: Lengths of the branches of the simulated fiber.
    :type length: list of strictly positive float or int

    :param thickness: Thickness of the generated fiber (default is 1).
    :type thickness: strictly positive float or int

    :param shift: Shift of the generated fiber toward a direction or another.
    :type shift: float or int

    :return: A 2D image of the simulated fiber without acquisition
    deterioration.
    :rtype: numpy.ndarray with shape (np.unique(pattern).size, imshape)
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Create a centered coordinate system
    x, y = np.meshgrid(range(-imshape[0]//2, imshape[0]//2),
                       range(-imshape[1]//2, imshape[1]//2))

    # Use distance maps to handle segments' localizations
    distance_to_line = np.abs(rho - (x * cos_theta + y * sin_theta))
    distance_on_line = (x * np.cos(theta + np.pi/2.0) +
                        y * np.sin(theta + np.pi/2.0)) + shift

    select_line = distance_to_line < thickness

    # Compute the branches points from length
    length = np.array(length)
    points = np.append([0], length.cumsum()) - length.sum() / 2.0

    # Compute simulated image (in multiple channels)
    full_shape = list(imshape)
    full_shape.insert(0, np.unique(pattern).size)
    fiber_image = np.zeros(full_shape)

    for index, channel in enumerate(pattern):
        select_branch = np.bitwise_and(
            distance_on_line >= points[index],
            distance_on_line <= points[index+1])
        select_segments = np.bitwise_and(select_line, select_branch)
        fiber_image[channel, select_segments] = 1.0

    return fiber_image


def fibers(thetas, rhos, imshape, thicknesses, patterns, lengths, shifts):
    """
    Simulate straight fibers images (with no acquisition deterioration).

    .. seealso:: dfa.simulate.fiber

    :param thetas: Angles of the Hesse normal form parametrization.
    :type thetas: list of floats between -pi/2 and pi/2

    :param rhos: Distances to origin of the Hesse normal form parametrization.
    :type rhos: list of float, list of int

    :param imshape: Shape of the generated image of fiber (2D image only).
    :type imshape: tuple of int with 2 elements

    :param thicknesses: Thickness of the generated fibers.
    :type thicknesses: list of strictly positive float or int

    :param patterns: Channel patterns of the simulated fibers.
    :type patterns: list of lists of int with same size as matching length

    :param lengths: Lengths of the generated fibers.
    :type lengths: list of lists of strictly positive float or int

    :param shifts: Shifts of the generated fibers toward a direction or another.
    :type shifts: list of float, list of int

    :return: A list of 2D images of the simulated fibers without acquisition
    deterioration.
    :rtype: list of numpy.ndarray with shape imshape
    """
    fiber_images = []

    for theta, rho, thickness, \
        pattern, length, shift in zip(thetas, rhos, thicknesses,
                                      patterns, lengths, shifts):
        fiber_images.append(fiber(
            theta, rho, imshape, pattern, length, thickness, shift))

    return fiber_images


def rfibers(imshape, number, theta_range, rho_range, thickness_range,
            model, shift_range):
    """
    Randomly simulate straight fibers images (with no acquisition
    deterioration).

    .. seealso:: dfa.simulate.fiber, dfa.simulate.fibers

    :param imshape: Shape of the generated image of fiber (2D image only).
    :type imshape: tuple of int with 2 elements

    :param number: Number of fibers to simulate randomly.
    :type number: strictly positive integer

    :param theta_range: Angle rangle of the Hesse normal form parametrization.
    :type theta_range: list or tuple with 2 elements

    :param rho_range: Distance range to origin of the Hesse normal form
    parametrization.
    :type rho_range: list or tuple with 2 elements

    :param thickness_range: Thickness range of fibers to simulate.
    :type thickness_range: list or tuple with 2 elements

    :param model: Model to use for simulations.
    :type model: dfa.model.Model

    :param shift_range: Shift range of fibers to simulate toward a direction
    or another.
    :type shift_range: list or tuple with 2 elements

    :return: A list of 2D images of the simulated fibers without acquisition
    deterioration.
    :rtype: list of numpy.ndarray with shapes imshape
    """

    patterns, lengths = model.simulate_patterns(number)

    return fibers((np.abs(np.diff(theta_range)) * np.random.rand(number) +
                   np.min(theta_range)).tolist(),
                  (np.abs(np.diff(rho_range)) * np.random.rand(number) +
                   np.min(rho_range)).tolist(),
                  imshape,
                  (np.abs(np.diff(thickness_range)) * np.random.rand(number) +
                   np.min(thickness_range)).tolist(),
                  patterns,
                  lengths,
                  (np.abs(np.diff(shift_range)) * np.random.rand(number) +
                   np.min(shift_range)).tolist())


def diffraction(input_image, psf, pos=0):
    """
    Simulate an out of focus effect on a single section with the specified PSF.

    Convolve the 2D input image with the 3D PSF at the desired position in order
    to simulate out of focus effect. When an object is not in the focal plane,
    its contribution to the focal plane is approximately equals to its
    convolution with the PSF at the given Z-position.

    The convolution is done in Fourier space, in order to speed up the
    computation.

    :param input_image: Input single section image.
    :type input_image: numpy.ndarray with 2 space dimensions and 1 for channels

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray with 3 dimensions

    :param pos: Relative position of input single section against the center of
    the PSF (default is 0, i.e. in focal plane).
    :type pos: int between -psf.shape[0]//2 and psf.shape[0]//2

    :return: The input single section with the requested out of focus.
    :rtype: numpy.ndarray with same shape as input_image
    """
    output_image = np.zeros(input_image.shape)

    for channel in range(input_image.shape[0]):
        output_image[channel, :, :] = sc.signal.fftconvolve(
            input_image[channel, :, :],
            psf[psf.shape[0] // 2 - pos, :, :],
            mode='same')

    return output_image


def photon_noise(input_image):
    """
    Simulate photon noise on input noise-free image.

    :param input_image: Input noise-free image.
    :type input_image: numpy.ndarray with 2 dimensions

    :return: Image corrupted with photon-noise (Poisson distribution).
    :rtype: numpy.ndarray with same shape as input_image
    """
    return np.random.poisson(input_image)


def image(fiber_objects, zindices, psf, outshape=None, snr=50):
    """
    Simulate image acquisition conditions of fiber objects.

    .. seealso:: dfa.simulation.diffraction, dfa.simulation.photon_noise

    The fiber objects need to be first simulated using the appropriate
    functions. With each object is associated a z-index giving the relative
    position of the fiber object plane to the focal plane of the final image.

    :param fiber_objects: Input simulated fiber objects.
    :type fiber_objects: list of numpy.ndarray with 2 dimensions

    :param zindices: Plane positions of fibers relative to the focal plane.
    :type zindices: list of int (or iterable)

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray with 3 dimensions

    :param outshape: Output shape of final image, i.e. quantization (default is
    the same as input).
    :type outshape: list of int or tuple of int

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray with 2 dimensions and shape outshape
    """
    if outshape is None:
        outshape = fiber_objects[0].shape

    final = np.zeros(fiber_objects[0].shape)
    zindices = np.array(zindices)
    fiber_objects = np.array(fiber_objects)

    # The most in-focus section is used to normalize intensities
    most_centered_zindex = np.abs(zindices)
    most_centered_zindex.sort()
    most_centered_zindex = most_centered_zindex[0]
    normalizing_constant = 1.0

    # Group fibers on the same z-index to speed-up the process
    for zindex in np.unique(zindices):
        current_section = np.zeros(final.shape)

        for fiber_object in fiber_objects[zindex == zindices]:
            current_section += fiber_object

        diffracted_image = diffraction(current_section, psf, zindex)
        final += diffracted_image

        if most_centered_zindex == zindex:
            normalizing_constant = diffracted_image.max()

    final /= normalizing_constant
    final[final < 0] = 0

    # When poisson noise, we have parameter lambda = 10^(SNR_dB / 5)
    final = np.round(final * np.power(10, snr/5))

    full_shape = list(outshape)
    full_shape.insert(0, final.shape[0])

    return photon_noise(ski.transform.resize(final, full_shape).astype(int))


def rimage(fiber_objects, zindex_range, psf, outshape=None, snr=50):
    """
    Simulate image acquisition conditions of fiber objects with random
    out-of-focus effects.

    .. seealso:: dfa.simulation.image

    :param fiber_objects: Input simulated fiber objects.
    :type fiber_objects: list of numpy.ndarray with 2 dimensions

    :param zindex_range: Plane positions range of fibers relative to the
    focal plane.
    :type zindex_range: tuple of int

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray with 3 dimensions

    :param outshape: Output shape of final image, i.e. quantization (default is
    the same as input).
    :type outshape: list of int or tuple of int

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray with 2 dimensions and shape outshape
    """
    if outshape is None:
        outshape = fiber_objects[0].shape

    zindices = np.random.randint(min(zindex_range), max(zindex_range),
                                 size=len(fiber_objects))

    return image(fiber_objects, zindices.tolist(), psf, outshape, snr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for saving simulation '
                             '(default: None).')

    fibers_group = parser.add_argument_group('Fibers')
    fibers_group.add_argument('--number', type=int, default=30,
                              help='Number of fiber segments to simulate '
                                   '(default: 30).')
    fibers_group.add_argument('--orientation', type=float, nargs=2,
                              default=[-0.3, -0.5],
                              help='Orientation range of fibers '
                                   '(default: [-0.3, -0.5]).')
    fibers_group.add_argument('--thickness', type=float, nargs=2,
                              default=[2, 3],
                              help='Thickness range of fibers '
                                   '(default: [2, 3]).')
    fibers_group.add_argument('--model', type=str, default=None,
                              help='Path to model file.')
    fibers_group.add_argument('--location', type=float, nargs=2,
                              default=[-500, 500],
                              help='Coordinates range of fiber center '
                                   '(default: [-500, 500]).')

    image_group = parser.add_argument_group('Image degradations')
    image_group.add_argument('psf_file', type=str, default=None,
                             help='Path to 3D PSF file.')
    image_group.add_argument('--shape', type=int, nargs=2, default=[512, 512],
                             help='Resolution of image output '
                                  '(default: [512, 512]).')
    image_group.add_argument('--z_index', type=int, nargs=2, default=[-15, 15],
                             help='Z-index of fiber objects '
                                  '(default: [-15, 15]).')
    image_group.add_argument('--snr', type=float, default=5,
                             help='SNR in decibels (default: 5).')

    args = parser.parse_args()

    if args.model is None:
        args.model = model.standard
    else:
        args.model = model.Model.load(args.model)

    from skimage import io

    simulated_psf = ski.io.imread(args.psf_file)
    fibers_images = rfibers(imshape=(1024, 1024), number=args.number,
                            theta_range=args.orientation,
                            rho_range=args.location,
                            thickness_range=args.thickness,
                            model=args.model,
                            shift_range=args.location)
    degraded_image = rimage(fibers_images, zindex_range=args.z_index,
                            psf=simulated_psf, snr=args.snr,
                            outshape=args.shape)

    if args.output is None:
        ski.io.imshow(degraded_image, cmap='gray')
        ski.io.show()
    else:
        ski.io.imsave(args.output, degraded_image.astype('int8'))

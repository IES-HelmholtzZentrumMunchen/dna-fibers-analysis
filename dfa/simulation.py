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


def fiber(theta, rho, imshape, thickness=1.0, length=100, shift=0):
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
    :type imshape: tuple of ints

    :param thickness: Thickness of the generated fiber (default is 1).
    :type thickness: strictly positive float or int

    :param length: Length of the generated fiber (default is 100).
    :type length: strictly positive float or int

    :param shift: Shift of the generated fiber toward a direction or another.
    :type shift: float or int

    :return: A 2D image of the simulated fiber without acquisition
    deterioration.
    :rtype: numpy.ndarray
    """
    assert type(theta) == float
    assert theta >= -np.pi/2
    assert theta < np.pi/2

    assert type(rho) == float or type(rho) == int

    assert type(imshape) == tuple
    assert all(type(n) == int for n in imshape)
    assert len(imshape) == 2

    assert type(thickness) == int or type(thickness) == float
    assert thickness > 0

    assert type(length) == int or type(length) == float
    assert length > 0

    assert type(shift) == int or type(shift) == float

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x, y = np.meshgrid(range(-imshape[0]//2, imshape[0]//2),
                       range(-imshape[1]//2, imshape[1]//2))

    distanceto_line = np.abs(rho - (x*cos_theta + y*sin_theta))
    distanceto_linecenter = np.sqrt(
        np.power(x - (-shift*sin_theta + rho*cos_theta), 2) +
        np.power(y - (shift*cos_theta + rho*sin_theta), 2))
    select_points = np.bitwise_and(distanceto_line < thickness,
                                   distanceto_linecenter < length)

    fiber_image = np.zeros(imshape)
    fiber_image[select_points] = 1.0

    return fiber_image


def fibers(thetas, rhos, imshape, thickness, lengths, shifts):
    """
    Simulate straight fibers images (with no acquisition deterioration).

    .. seealso:: dfa.simulate.fiber

    :param thetas: Angles of the Hesse normal form parametrization.
    :type thetas: list of floats between -pi/2 and pi/2

    :param rhos: Distances to origin of the Hesse normal form parametrization.
    :type rhos: list float or int

    :param imshape: Shape of the generated image of fiber (2D image only).
    :type imshape: tuple of ints

    :param thickness: Thickness of the generated fibers.
    :type thickness: list of strictly positive float or int

    :param lengths: Lengths of the generated fibers.
    :type lengths: list of strictly positive float or int

    :param shifts: Shifts of the generated fibers toward a direction or another.
    :type shifts: list of floats or ints

    :return: A list of 2D images of the simulated fibers without acquisition
    deterioration.
    :rtype: list of numpy.ndarray
    """
    assert type(thetas) == list
    assert type(rhos) == list

    assert type(imshape) == tuple
    assert all(type(n) == int for n in imshape)
    assert len(imshape) == 2

    assert type(thickness) == list
    assert type(lengths) == list
    assert type(shifts) == list

    fiber_images = []

    for theta, rho, thicknes, length, shift in zip(thetas, rhos, thickness,
                                                   lengths, shifts):
        fiber_images.append(
            fiber(theta, rho, imshape, thicknes, length, shift))

    return fiber_images


def rfibers(imshape, number, theta_range, rho_range, thickness_range,
            length_range, shift_range):
    """
    Randomely simulate straight fibers images (with no acquisition
    deterioration).

    .. seealso:: dfa.simulate.fiber, dfa.simulate.fibers

    :param imshape: Shape of the generated image of fiber (2D image only).
    :type imshape: tuple of ints

    :param number: Number of fibers to simulate randomely.
    :type number: strictly positive integer

    :param theta_range: Angle rangle of the Hesse normal form parametrization.
    :type theta_range: list or tuple with 2 elements.

    :param rho_range: Distance range to origin of the Hesse normal form
    parametrization.
    :type rho_range: list or tuple with 2 elements.

    :param thickness_range: Thickness range of fibers to simulate.
    :type thickness_range: list or tuple with 2 elements

    :param length_range: Length range of fibers to simulate.
    :type length_range: list or tuple with 2 elements.

    :param shift_range: Shift range of fibers to simulate toward a direction
    or another.
    :type shift_range: list or tuple with 2 elements

    :return: A list of 2D images of the simulated fibers without acquisition
    deterioration.
    :rtype: list of numpy.ndarray
    """
    assert type(imshape) == tuple
    assert all(type(n) == int for n in imshape)
    assert len(imshape) == 2

    assert type(number) == int
    assert number > 0

    assert type(theta_range) == list or type(theta_range) == tuple
    assert len(theta_range) == 2

    assert type(rho_range) == list or type(rho_range) == tuple
    assert len(rho_range) == 2

    assert type(thickness_range) == list or type(thickness_range) == tuple
    assert len(thickness_range) == 2

    assert type(length_range) == list or type(length_range) == tuple
    assert len(length_range) == 2

    assert type(shift_range) == list or type(shift_range) == tuple
    assert len(shift_range) == 2

    return fibers(
        (np.abs(np.diff(theta_range)) * np.random.rand(number) + np.min(
            theta_range)).tolist(),
        (np.abs(np.diff(rho_range)) * np.random.rand(number) + np.min(
            rho_range)).tolist(),
        imshape,
        (np.abs(np.diff(thickness_range)) * np.random.rand(number) + np.min(
            thickness_range)).tolist(),
        (np.abs(np.diff(length_range)) * np.random.rand(number) + np.min(
            length_range)).tolist(),
        (np.abs(np.diff(shift_range)) * np.random.rand(number) + np.min(
            shift_range)).tolist())


def focus(input_image, psf, pos=0):
    """
    Simulate an out of focus effect on a single section with the specified PSF.

    Convolve the 2D input image with the 3D PSF at the desired position in order
    to simulate out of focus effect. When an object is not in the focal plane,
    its contribution to the focal plane is approximately equals to its
    convolution with the PSF at the given Z-position.

    The convolution is done in Fourier space, in order to speed up the
    computation.

    :param input_image: Input single section image.
    :type input_image: numpy.ndarray

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray

    :param pos: Relative position of input single section against the center of
    the PSF (default is 0, i.e. in focal plane).
    :type pos: int

    :return: The input single section with the requested out of focus.
    :rtype: numpy.ndarray
    """
    assert type(input_image) == np.ndarray
    assert len(input_image.shape) == 2

    assert type(psf) == np.ndarray
    assert len(psf.shape) == 3

    assert type(pos) == int
    assert pos <= psf.shape[0]//2
    assert pos >= -psf.shape[0]//2

    return sc.signal.fftconvolve(input_image,
                                 psf[psf.shape[0] // 2 - pos, :, :],
                                 mode='same')


def pnoise(input_image):
    """
    Simulate photon noise on input noise-free image.

    :param input_image: Input noise-free image.
    :type input_image: numpy.ndarray

    :return: Image corrupted with photon-noise (Poisson distribution).
    :rtype: numpy.ndarray
    """
    assert type(input_image) == np.ndarray

    return np.random.poisson(input_image)


def image(fiber_objects, zindices, psf, outshape=None, snr=50):
    """
    Simulate image acquisition conditions of fiber objects.

    The fiber objects need to be first simulated using the appropriate
    functions. With each object is associated a z-index giving the relative
    position of the fiber object plane to the focal plane of the final image.

    :param fiber_objects: Input simulated fiber objects.
    :type fiber_objects: list of numpy.ndarray

    :param zindices: Plane positions of fibers relative to the focal plane.
    :type zindices: list of ints or Iteratable

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray

    :param outshape: Output shape of final image, i.e. quantization (default is
    the same as input).
    :type outshape: list or tuple

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray
    """
    assert type(fiber_objects) == list
    assert all(type(fiber_object) == np.ndarray
               for fiber_object in fiber_objects)

    assert type(zindices) == list
    assert all(type(zindex) == int for zindex in zindices)

    assert type(psf) == np.ndarray
    assert len(psf.shape) == 3

    if outshape is None:
        outshape = fiber_objects[0].shape

    assert type(outshape) == list or type(outshape) == tuple
    assert len(outshape) == 2

    assert type(snr) == float or type(snr) == int

    final = np.zeros(fiber_objects[0].shape)

    for fiber_object, zindex in zip(fiber_objects, zindices):
        # When poisson noise, we have lambda = 10^(SNR_dB / 5)
        final += focus(fiber_object*np.power(10, snr/5), psf, zindex)

    final[final < 0] = 0

    # FIXME regarder le comportement de cette fonction (gestion intensité)
    return pnoise(ski.transform.resize(final, outshape).astype(int))


def rimage(fiber_objects, zindex_range, psf, outshape=None, snr=50):
    """
    Simulate image acquisition conditions of fiber objects with random
    out-of-focus effects.

    :param fiber_objects: Input simulated fiber objects.
    :type fiber_objects: list of numpy.ndarray

    :param zindex_range: Plane positions range of fibers relative to the
    focal plane.
    :type zindex_range: tuple of ints

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray

    :param outshape: Output shape of final image, i.e. quantization (default is
    the same as input).
    :type outshape: list or tuple

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray
    """
    assert type(fiber_objects) == list
    assert all(type(fiber_object) == np.ndarray
               for fiber_object in fiber_objects)

    assert type(zindex_range) == list or type(zindex_range) == tuple
    assert len(zindex_range) == 2

    assert type(psf) == np.ndarray
    assert len(psf.shape) == 3

    if outshape is None:
        outshape = fiber_objects[0].shape

    assert type(outshape) == list or type(outshape) == tuple
    assert len(outshape) == 2

    assert type(snr) == float or type(snr) == int

    zindices = np.random.randint(min(zindex_range), max(zindex_range),
                                 size=len(fiber_objects))

    return image(fiber_objects, zindices.tolist(), psf, outshape, snr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for saving simulation.')
    fibers_group = parser.add_argument_group('Fibers')
    fibers_group.add_argument('--number', type=int, default=30,
                              help='Number of fiber segments to simulate.')
    fibers_group.add_argument('--orientation', type=float, nargs=2,
                              default=[-0.3, -0.5],
                              help='Orientation range of fibers.')
    fibers_group.add_argument('--thickness', type=float, nargs=2,
                              default=[2, 3],
                              help='Thickness range of fibers.')
    fibers_group.add_argument('--length', type=float, nargs=2,
                              default=[50, 200],
                              help='Total length range of fibers.')
    fibers_group.add_argument('--location', type=float, nargs=2,
                              default=[-500, 500],
                              help='Coordinates range of fiber center.')
    image_group = parser.add_argument_group('Image degradations')
    image_group.add_argument('psf_file', type=str, default=None,
                             help='Path to 3D PSF file.')
    image_group.add_argument('--shape', type=int, nargs=2, default=[512, 512],
                             help='Resolution of image output.')
    image_group.add_argument('--z_index', type=int, nargs=2, default=[-10, 10],
                             help='Z-index of fiber objects.')
    image_group.add_argument('--snr', type=float, default=0,
                             help='SNR in decibels.')
    args = parser.parse_args()

    from skimage import io

    simulated_psf = ski.io.imread(args.psf_file)
    fibers_images = rfibers(imshape=(1024, 1024), number=args.number,
                            theta_range=args.orientation,
                            rho_range=args.location,
                            thickness_range=args.thickness,
                            length_range=args.length,
                            shift_range=args.location)
    degraded_image = rimage(fibers_images, zindex_range=args.z_index,
                            psf=simulated_psf, snr=args.snr, outshape=args.shape)
    ski.io.imshow(degraded_image, cmap='gray')
    ski.io.show()

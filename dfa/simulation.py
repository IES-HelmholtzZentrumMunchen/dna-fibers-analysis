"""
Simulation module of the DNA fiber analysis package.

Use this module to simulate synthetic images of DNA fibers. Both fiber
segments and image degradations can be controlled.
"""
from debtcollector import removals

import numpy as np
import scipy as sc
from scipy import signal

from scipy.interpolate import splprep, splev

import skimage as ski
from skimage import measure
from skimage import io

from dfa import modeling


@removals.remove
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

    select_line = distance_to_line < thickness/2.0

    # Compute the branches points from length
    length = np.array(length)
    points = np.append([0], length.cumsum()) - length.sum() / 2.0

    # Compute simulated image (in multiple channels)
    full_shape = list(imshape)
    full_shape.insert(0, 2)
    fiber_image = np.zeros(full_shape)

    for index, channel in enumerate(pattern):
        select_branch = np.bitwise_and(
            distance_on_line >= points[index],
            distance_on_line <= points[index+1])
        select_segments = np.bitwise_and(select_line, select_branch)
        fiber_image[channel, select_segments] = 1.0

    return fiber_image


def fiber_spline(angle, length, shift=(0, 0), step=4, interp_step=1,
                 perturbations_force=0.5, bending_elasticity=2,
                 bending_force=2):
    """
    Simulate a fiber path object with local and global perturbations.

    :param angle: Orientation angle of the fiber to simulate (in degrees).
    :type angle: float

    :param length: Length of the fiber to simulate (in pixels.
    :type length: float

    :param shift: Translation vector of the fiber from image's center
    (in pixels, default is [0, 0]).
    :type shift: np.ndarray

    :param step: Step size of control points along path (in pixels,
    default is 4).
    :type step: float

    :param interp_step: Step size of sampled points along path (in pixels,
    default is 1).
    :type interp_step: float

    :param perturbations_force: Force of the local perturbations along the
    fiber path to simulate (range is ]0,+inf[, default is 0.5).
    :type perturbations_force: float

    :param bending_elasticity: Elasticity of the global perturbation bending
    the fiber path to be simulated (range is ]0,+inf[, default is 2).
    :type bending_elasticity: float

    :param bending_force: Force of the global perturbation bending the fiber
    path to be simulated (range is ]0+inf[, default is 0.5).
    :type bending_force: float

    :return:
    """
    # Find limit points of the fiber
    radian_angle = np.pi * angle / 180.0
    vx, vy = np.cos(radian_angle), np.sin(radian_angle)
    c = length / 2.0
    a, b = -c * np.array([vx, vy]) + shift, c * np.array([vx, vy]) + shift

    # Create local perturbations along the fiber path
    num_points = int(round(length/step))
    x = np.linspace(a[0], b[0], num=num_points) + \
        perturbations_force * np.random.randn(num_points)
    y = np.linspace(a[1], b[1], num=num_points) + \
        perturbations_force * np.random.randn(num_points)
    u = np.linspace(0, 1, num=num_points)

    # Create global perturbation along the fiber path
    d = np.exp(-0.5 * (u - 0.25 * np.random.randn() - 0.5)**2
               * bending_elasticity**2)
    bending_force *= (-1)**np.random.binomial(1, 0.5)
    x += -bending_force * vy * d + bending_force * vy
    y += bending_force * vx * d - bending_force * vx

    # Interpolate with B-Splines
    splines = splprep(np.stack((x, y)), s=0, u=u)
    xnew, ynew = splev(np.linspace(0, 1, num=int(round(length/interp_step))),
                       splines[0])

    return xnew, ynew


def fiber_inhomogeneity(num_of_points, local_force=0.05, global_force=1,
                        global_rate=1.5):
    """
    Simulate signal inhomogeneity in fiber.

    The intensity factors are simulated using local and global perturbations of
    a even illumination (every point with intensity set to 1).

    :param num_of_points: Number of points on the fiber.
    :type num_of_points: strictly positive int

    :param local_force: Force of local signal inhomogeneity (default is 0.05).
    :type local_force: float

    :param global_force: Force of the global signal inhomogeneity.
    :type global_force: float

    :param global_rate: Rate of modulation of the global signal inhomogeneity.
    :type global_rate: float

    :return: Sequence of inhomogeneity factors.
    :rtype: numpy.ndarray
    """
    t = np.arange(0, num_of_points)
    s = np.ones(num_of_points)

    # Create local inhomogeneity
    s += local_force * np.random.randn(num_of_points)

    # Create global inhomogeneity
    dd = np.abs(0.5 * (t.max()-t.min()) * np.random.randn())
    print(dd)
    global_rate /= num_of_points
    d = np.exp(-0.5 * (t - dd)**2
               * global_rate**2)
    s *= global_force * d

    return s


def fiber_disconnections(fiber_points, disc_prob=0.2, return_prob=0.5):
    """
    Simulate disconnections in fiber.

    The disconnections are modeled as a state of a Markov process. The points
    sampled on the fiber path define a Markov chain and those points can have
    two states: not disconnected or disconnected. The corresponding
    probabilities rule the random apparitions of the disconnections.

    :param fiber_points: Points of the input fiber.
    :type fiber_points: tuple of numpy.ndarray

    :param disc_prob: Probability to have disconnections (default is 0.2).
    :type disc_prob: float

    :param return_prob: Probability to stop the disconnection (default is 0.5).
    :type return_prob: float

    :return: Degraded points of the input fibers.
    """
    state = 1  # initial state is no disconnected
    select = []  # points with not disconnected state will be selected

    for _ in fiber_points[0]:
        if state == 1:
            select.append(1-np.random.binomial(1, disc_prob) == 1)
        else:
            select.append(np.random.binomial(1, return_prob) == 1)

    return fiber_points[0][select], fiber_points[1][select]


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
    :type model: dfa.modeling.Model

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


def shot_noise(input_image, snr):
    """
    Simulate photon noise on input noise-free image.

    :param input_image: Input noise-free image.
    :type input_image: numpy.ndarray with 2 dimensions

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Image corrupted with photon-noise (Poisson distribution).
    :rtype: numpy.ndarray with same shape as input_image
    """
    # When poisson noise, we have parameter lambda = 10^(SNR_dB / 5)
    return np.random.poisson(np.round(input_image *
                                      np.power(10, snr/5)).astype(int))


def image(fiber_objects, zindices, psf, binning=1, snr=20):
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

    :param binning: Binning of output final image, i.e. quantization (default is
    the same as input).
    :type binning: strictly positive int

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray with 2 dimensions and shape outshape
    """
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

    # We should do the binning first and then the shot noise. But since
    # a sum of Poisson random variables is also a Poisson random variable,
    # the results is formally equivalent as doing the binning last.
    return ski.measure.block_reduce(shot_noise(final, snr),
                                    (1, binning, binning), func=np.sum)


def rimage(fiber_objects, zindex_range, psf, binning=1, snr=10):
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

    :param binning: Binning of output final image, i.e. quantization (default is
    the same as input).
    :type binning: strictly positive int

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray with 2 dimensions and shape outshape
    """
    zindices = np.random.randint(min(zindex_range), max(zindex_range),
                                 size=len(fiber_objects))

    return image(fiber_objects, zindices.tolist(), psf, binning, snr)


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
                              default=[5, 6],
                              help='Thickness range of fibers '
                                   '(default: [5, 6]).')
    fibers_group.add_argument('--model', type=str, default=None,
                              help='Path to model file (default: '
                                   'internal standard).')
    fibers_group.add_argument('--location', type=float, nargs=2,
                              default=[-0.5, 0.5],
                              help='Coordinates range of fiber center '
                                   '(default: [-0.5, 0.5]).')

    image_group = parser.add_argument_group('Image degradations')
    image_group.add_argument('psf_file', type=str, default=None,
                             help='Path to 3D PSF file.')
    image_group.add_argument('--binning', type=int, default=2,
                             help='Binning of image output (default: 2).')
    image_group.add_argument('--z_index', type=int, nargs=2, default=[-15, 15],
                             help='Z-index of fiber objects '
                                  '(default: [-15, 15]).')
    image_group.add_argument('--snr', type=float, default=5,
                             help='SNR in decibels (default: 5).')

    args = parser.parse_args()

    if args.model is None:
        args.model = modeling.standard
    else:
        args.model = modeling.Model.load(args.model)

    highres_shape = (1024, 1024)
    args.location[0] *= highres_shape[0]
    args.location[1] *= highres_shape[1]

    from skimage import io

    simulated_psf = ski.io.imread(args.psf_file)
    fibers_images = rfibers(imshape=highres_shape, number=args.number,
                            theta_range=args.orientation,
                            rho_range=args.location,
                            thickness_range=args.thickness,
                            model=args.model,
                            shift_range=args.location)
    degraded_image = rimage(fibers_images, zindex_range=args.z_index,
                            psf=simulated_psf, snr=args.snr,
                            binning=args.binning)

    if args.output is None:
        ski.io.imshow(degraded_image[0], cmap='gray')
        ski.io.show()
    else:
        ski.io.imsave(args.output, degraded_image.astype('int16'))

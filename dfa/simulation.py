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

    :return: The coordinates of the points as an (2,N) array.
    :rtype: numpy.ndarray
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

    return np.vstack((xnew, ynew))


def fiber_inhomogeneity(num_of_points, local_force=0.05, global_force=0.25,
                        global_rate=1.5):
    """
    Simulate signal inhomogeneity in fiber.

    The intensity factors are simulated using local and global perturbations of
    a even illumination (every point with intensity set to 1).

    :param num_of_points: Number of points on the fiber.
    :type num_of_points: strictly positive int

    :param local_force: Force of local signal inhomogeneity (default is 0.05).
    :type local_force: float

    :param global_force: Force of the global signal inhomogeneity (default is
    0.25).
    :type global_force: float

    :param global_rate: Rate of modulation of the global signal inhomogeneity
    (default is 1.5).
    :type global_rate: float

    :return: Sequence of inhomogeneity factors.
    :rtype: numpy.ndarray
    """
    t = np.arange(0, num_of_points)
    s = np.ones(num_of_points)

    # Create local inhomogeneity
    s += local_force * np.random.randn(num_of_points)

    # Create global inhomogeneity
    global_rate /= num_of_points
    d = np.exp(-0.5 * (t - np.abs(0.5 *
                                  (t.max()-t.min()) * np.random.randn()))**2
               * global_rate**2)
    s += global_force * d

    return s - s.min() + 1


def fiber_disconnections(fiber_points, disc_prob=0.2, return_prob=0.5):
    """
    Simulate disconnections in fiber.

    The disconnections are modeled as a state of a Markov process. The points
    sampled on the fiber path define a Markov chain and those points can have
    two states: not disconnected or disconnected. The corresponding
    probabilities rule the random apparitions of the disconnections.

    :param fiber_points: Points of the input fiber.
    :type fiber_points: numpy.ndarray

    :param disc_prob: Probability to have disconnections (default is 0.2).
    :type disc_prob: float

    :param return_prob: Probability to stop the disconnection (default is 0.5).
    :type return_prob: float

    :return: The coordinates of the degraded points as an (2,N) array.
    :rtype: numpy.ndarray
    """
    state = 1  # initial state is no disconnected
    select = []  # points with not disconnected state will be selected

    for _ in fiber_points[0]:
        if state == 1:
            select.append(1-np.random.binomial(1, disc_prob) == 1)
        else:
            select.append(np.random.binomial(1, return_prob) == 1)

    return fiber_points[:, select]


def fibers(geom_props, disc_props, signal_props):
    """
    Simulate fibers fluophores as points in image space.

    .. seealso:: dfa.simulation.fiber_spline,
    dfa.simulation.fiber_disconnections, dfa.simulation.fiber_inhomogeneity

    :param geom_props: Geometrical properties (see dfa.simulation.fiber_spline).
    :type geom_props: list of dict

    :param disc_props: Disconnections properties (see
    dfa.simulation.fiber_disconnections).
    :type disc_props: list of dict

    :param signal_props: Signal properties (see
    dfa.simulation.fiber_inhomogeneity).
    :type signal_props: list of dict

    :return: The points/signal of fibers paths simulated.
    :rtype: list of tuple
    """
    fiber_objects = []

    for geom_prop, disc_prop, signal_prop in \
            zip(geom_props, disc_props, signal_props):
        f = fiber_disconnections(fiber_spline(**geom_prop), **disc_prop)
        s = fiber_inhomogeneity(f.shape[1], **signal_prop)
        fiber_objects.append((f, s))

    return fiber_objects


def rfibers(number, angle_range, shift_range, perturbations_force_range,
            bending_elasticity_range, bending_force_range, disc_prob_range,
            return_prob_range, local_force_range, global_force_range,
            global_rate_range, model=modeling.standard):
    """
    Randomly simulate fibers objects with geometrical deterioration /
    signal inhomogeneity.

    .. seealso:: dfa.simulate.fibers

    For information abount the ranges, refer to dfa.simulation.fiber_spline,
    dfa.simulation.fiber_disconnections and
    dfa.simulation.inhomogeneity.
    """
    def _uniform_sample_within_range(sampling_range, sample_number=number):
        return np.abs(np.diff(sampling_range)) * \
               np.random.rand(sample_number) + \
               np.min(sampling_range)

    patterns, lengths = model.simulate_patterns(number)

    geom_props = []

    angles = _uniform_sample_within_range(angle_range)
    shifts_x = _uniform_sample_within_range(shift_range[0])
    shifts_y = _uniform_sample_within_range(shift_range[1])
    perturbations_forces = _uniform_sample_within_range(
        perturbations_force_range)
    bending_elasticities = _uniform_sample_within_range(
        bending_elasticity_range)
    bending_forces = _uniform_sample_within_range(bending_force_range)

    disc_props = []

    disc_probs = _uniform_sample_within_range(disc_prob_range)
    return_probs = _uniform_sample_within_range(return_prob_range)

    signal_props = []

    local_forces = _uniform_sample_within_range(local_force_range)
    global_forces = _uniform_sample_within_range(global_force_range)
    global_rates = _uniform_sample_within_range(global_rate_range)

    for i in range(number):
        geom_props.append({
            'angle': angles[i], 'length': sum(lengths[i]),
            'shift': (shifts_x[i], shifts_y[i]),
            'perturbations_force': perturbations_forces[i],
            'bending_elasticity': bending_elasticities[i],
            'bending_force': bending_forces[i]})
        disc_props.append({
            'disc_prob': disc_probs[i], 'return_prob': return_probs[i]})
        signal_props.append({
            'local_force': local_forces[i], 'global_force': global_forces[i],
            'global_rate': global_rates[i]})

    return fibers(geom_props, disc_props, signal_props)


def image_by_diffraction(shape, fibers_points, fibers_signal, psf,
                         positions=None):
    """
    Create a diffraction limited image from points along fiber paths.

    :param shape: Shape of the output image.
    :type shape: tuple of int

    :param fibers_points: Coordinates points in image space of the fiber paths.
    :type fibers_points: list of numpy.ndarray

    :param fibers_signal: Signal power of each points along fiber paths.
    :type fibers_signal: list of numpy.ndarray

    :param psf: 3D image of the PSF used for diffraction simulation.
    :type psf: numpy.ndarray

    :param positions: Positions in the PSF z-stack used to simulate
    out-of-focus.
    When set to None (default), the positions are 0 (in-focus).
    :type positions: list of int or None

    :return: Simulated diffraction-limited image.
    :rtype: numpy.ndarray
    """
    if positions is None:
        positions = [0] * len(fibers_points)

    # initialize output
    output_image = np.zeros(np.add(shape, psf.shape[:-1]))
    half_x = psf.shape[2] // 2
    half_y = psf.shape[1] // 2
    half_z = psf.shape[0] // 2

    # add diffracted dirac sources to image (psf at each fibers point)
    for fiber_points, fiber_signal, position in zip(fibers_points,
                                                    fibers_signal, positions):
        for x, y, signal in zip(fiber_points[0], fiber_points[1], fiber_signal):
            rx = round(x)
            ry = round(y)

            # print(rx, rx+psf.shape[2], ry, ry+psf.shape[1], psf.shape)

            if 0 <= rx < shape[1] and 0 <= ry < shape[0]:
                output_image[int(ry):int(ry) + psf.shape[1],
                             int(rx):int(rx) + psf.shape[2]] += \
                    signal * psf[half_z - position]

    return output_image[half_y:-half_y, half_x:-half_x]


@removals.remove
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


def image(fiber_objects, shape, zindices, psf, snr=20):
    """
    Simulate image acquisition conditions of fiber objects.

    .. seealso:: dfa.simulation.diffraction, dfa.simulation.photon_noise

    The fiber objects need to be first simulated using the appropriate
    functions. With each object is associated a z-index giving the relative
    position of the fiber object plane to the focal plane of the final image.

    :param fiber_objects: Input simulated fiber objects.
    :type fiber_objects: list of numpy.ndarray with 2 dimensions

    :param shape: Shape of the output image.
    :type shape: tuple of int

    :param zindices: Plane positions of fibers relative to the focal plane.
    :type zindices: list of int (or iterable)

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray with 3 dimensions

    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray with 2 dimensions and shape outshape
    """
    return shot_noise(image_by_diffraction(
        shape, *zip(*fiber_objects), psf, zindices)+1, snr)


def rimage(fiber_objects, shape, zindex_range, psf, snr=10):
    """
    Simulate image acquisition conditions of fiber objects with random
    out-of-focus effects.

    .. seealso:: dfa.simulation.image

    :param fiber_objects: Input simulated fiber objects.
    :type fiber_objects: list of numpy.ndarray with 2 dimensions

    :param shape: Shape of the output image.
    :type shape: tuple of int

    :param zindex_range: Plane positions range of fibers relative to the
    focal plane.
    :type zindex_range: tuple of int

    :param psf: PSF used to simulate the microscope's diffraction of light.
    :type psf: numpy.ndarray with 3 dimensions
f
    :param snr: Signal-to-noise ratio of the simulated image (in dB).
    :type snr: float

    :return: Final simulated image of fibers with acquisition artefacts.
    :rtype: numpy.ndarray with 2 dimensions and shape outshape
    """
    if zindex_range is None:
        zindices = None
    else:
        zindices = np.random.randint(min(zindex_range), max(zindex_range),
                                     size=len(fiber_objects)).tolist()

    return image(fiber_objects, shape, zindices, psf, snr)


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

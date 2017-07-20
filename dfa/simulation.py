"""
Simulation module of the DNA fiber analysis package.

Use this module to simulate synthetic images of DNA fibers. Both fiber
segments and image degradations can be controlled.
"""
import numpy as np

from scipy.interpolate import splprep, splev

from dfa import modeling


def fiber(angle, length, shift=(0, 0), step=4, interp_step=1,
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


def fiber_inhomogeneity(num_of_points, number_of_channels, pattern, length,
                        local_force=0.05, global_force=0.25, global_rate=1.5):
    """
    Simulate signal inhomogeneity in fiber.

    The intensity factors are simulated using local and global perturbations of
    a even illumination (every point with intensity set to 1).

    :param num_of_points: Number of points on the fiber.
    :type num_of_points: strictly positive int

    :param number_of_channels: Number of channels of the output image.
    :type number_of_channels: int

    :param pattern: The input patterns of the fibers to simulate.
    :type pattern: list of int

    :param length: The lengths of the input pattern segments to simulate.
    :type length: list of float

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
    u = t / t.max()
    length_cumsum = np.cumsum([0.] + length) / sum(length)
    s = np.zeros((number_of_channels, num_of_points))

    # create global inhomogeneity
    global_rate /= num_of_points
    d = np.exp(-0.5 * (t - np.abs(0.5 *
                                  (t.max()-t.min()) * np.random.randn()))**2
               * global_rate**2)

    for i in range(len(s)):
        # create channel signal (following pattern)
        for j in range(len(pattern)):
            if pattern[j] == i:
                s[i, np.bitwise_and(length_cumsum[j] <= u,
                                    u <= length_cumsum[j+1])] = 1

        # create local inhomogeneity
        s[i] += local_force * np.random.randn(num_of_points)

        s[i] += global_force * d
        s[i] -= s[i].min()

    return s


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


def fibers(number_of_channels, patterns, lengths, geom_props, disc_props,
           signal_props):
    """
    Simulate fibers fluophores as points in image space.

    .. seealso:: dfa.simulation.fiber_spline,
    dfa.simulation.fiber_disconnections, dfa.simulation.fiber_inhomogeneity

    :param number_of_channels: Number of channels of the output image.
    :type number_of_channels: int

    :param patterns: The input patterns of the fibers to simulate.
    :type patterns: list of list of int

    :param lengths: The lengths of the input pattern segments to simulate.
    :type lengths: list of list of float

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

    for pattern, length, geom_prop, disc_prop, signal_prop in \
            zip(patterns, lengths, geom_props, disc_props, signal_props):
        f = fiber_disconnections(fiber(**geom_prop), **disc_prop)
        s = fiber_inhomogeneity(f.shape[1], number_of_channels,
                                pattern, length, **signal_prop)
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
    number_of_channels = len(model.channels_names)

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

    return fibers(number_of_channels, patterns, lengths,
                  geom_props, disc_props, signal_props)


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
    output_image = np.zeros((len(fibers_signal[0]),) +
                            tuple(np.add(shape, psf.shape[:-1])))
    half_x = psf.shape[2] // 2
    half_y = psf.shape[1] // 2
    half_z = psf.shape[0] // 2

    # add diffracted dirac sources to image (psf at each fibers point)
    for fiber_points, fiber_signal, position in zip(fibers_points,
                                                    fibers_signal, positions):
        for i in range(len(fiber_signal)):
            for x, y, signal \
                    in zip(fiber_points[0], fiber_points[1], fiber_signal[i]):
                rx = round(x)
                ry = round(y)

                if 0 <= rx < shape[1] and 0 <= ry < shape[0]:
                    output_image[i,
                                 int(ry):int(ry) + psf.shape[1],
                                 int(rx):int(rx) + psf.shape[2]] += \
                        signal * psf[half_z - position]

    # normalize
    for i in range(output_image.shape[0]):
        if output_image[i].max() > 0:
            output_image[i] /= output_image[i].max()

    return output_image[:, half_y:-half_y, half_x:-half_x]


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
    for i in range(input_image.shape[0]):
        input_image[i] = np.round(input_image[i] * np.power(10, snr / 5))
        background = np.equal(input_image[i], 0)
        input_image[i, background] = np.random.rand(
            input_image[i, background].size)
        input_image[i] = np.random.poisson(
            np.round(input_image[i]).astype(int))

    return input_image


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
    clean_image = image_by_diffraction(
        shape, *zip(*fiber_objects), psf, zindices)

    return shot_noise(clean_image, snr)


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
                              default=[30, 40],
                              help='Orientation range of fibers '
                                   '(in degrees, default: [30, 40]).')
    fibers_group.add_argument('--model', type=str, default=None,
                              help='Path to model file (default: '
                                   'internal standard).')
    fibers_group.add_argument('--location', type=float, nargs=4,
                              default=[100, 924, 100, 924],
                              help='Coordinates range of fiber center '
                                   '(<x_min> <x_max> <y_min> <y_max>, '
                                   'in pixels, default: '
                                   '[100, 924, 100, 924]).')
    fibers_group.add_argument('--perturbations-force-range', type=float,
                              nargs=2, default=[0.1, 0.3],
                              help='Local perturbations of the fiber path '
                                   '(default is [0.1, 0.2]).')
    fibers_group.add_argument('--bending-elasticity-range', type=float,
                              nargs=2, default=[1, 3],
                              help='Elasticity of the global bending of the '
                                   'fiber path (default is [1, 3]).')
    fibers_group.add_argument('--bending-force-range', type=float,
                              nargs=2, default=[10, 20],
                              help='Force of the global bending of the fiber '
                                   'path (default is [10, 20]).')
    fibers_group.add_argument('--disconnection-probability-range', type=float,
                              nargs=2, default=[0.05, 0.1],
                              help='Probability of disconnections happening '
                                   'on the fiber path (default is '
                                   '[0.05, 0.1]).')
    fibers_group.add_argument('--return-probability-range', type=float,
                              nargs=2, default=[0.5, 0.7],
                              help='Probability of recovering when the fiber '
                                   'path is disconnected (default is '
                                   '[0.5, 0.7]).')
    fibers_group.add_argument('--local-force-range', type=float, nargs=2,
                              default=[0.05, 0.2],
                              help='Force of local signal inhomogeneity '
                                   '(default is [0.05, 0.2]).')
    fibers_group.add_argument('--global-force-range', type=float, nargs=2,
                              default=[2, 5],
                              help='Force of global signal inhomogeneity '
                                   '(default is [2, 5]).')
    fibers_group.add_argument('--global-rate-range', type=float, nargs=2,
                              default=[0.5, 1.5],
                              help='Rate of global signal inhomogeneity '
                                   '(default is [0.5, 1.5]).')

    image_group = parser.add_argument_group('Image degradations')
    image_group.add_argument('--shape', type=int, nargs=2, default=[1024, 1024],
                             help='Shape of the output image (default is'
                                  '[1024, 1024]). Note that the pixel size'
                                  'is the same as the PSF pixel size.')
    image_group.add_argument('psf_file', type=str, default=None,
                             help='Path to 3D PSF file.')
    image_group.add_argument('--z_index', type=int, nargs=2, default=[-1, 1],
                             help='Z-index of fiber objects '
                                  '(default: [-1, 1]).')
    image_group.add_argument('--snr', type=float, default=7,
                             help='SNR in decibels (default: 7).')

    args = parser.parse_args()

    if args.model is None:
        args.model = modeling.standard
    else:
        args.model = modeling.Model.load(args.model)

    from skimage import io

    simulated_psf = io.imread(args.psf_file)

    fibers_objects = rfibers(
        number=args.number, angle_range=args.orientation,
        shift_range=[tuple(args.location[:2]), tuple(args.location[2:])],
        perturbations_force_range=args.perturbations_force_range,
        bending_elasticity_range=args.bending_elasticity_range,
        bending_force_range=args.bending_force_range,
        disc_prob_range=args.disconnection_probability_range,
        return_prob_range=args.return_probability_range,
        local_force_range=args.local_force_range,
        global_force_range=args.global_force_range,
        global_rate_range=args.global_rate_range, model=args.model)

    degraded_image = rimage(
        fiber_objects=fibers_objects, shape=args.shape,
        zindex_range=args.z_index, psf=simulated_psf, snr=args.snr)

    if args.output is None:
        io.imshow_collection([*degraded_image], cmap='gray')
        io.show()
    else:
        io.imsave(args.output, degraded_image.astype('int16'))

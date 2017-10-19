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

    Parameters
    ----------
    angle : float
        Orientation angle of the fiber to simulate (in degrees).

    length : float
        Length of the fiber to simulate (in pixels.

    shift : numpy.ndarray
        Translation vector of the fiber from image's center (in pixels,
        default is [0, 0]).

    step : float
        Step size of control points along path (in pixels, default is 4).

    interp_step : float
        Step size of sampled points along path (in pixels, default is 1).

    perturbations_force : float
        Force of the local perturbations along the fiber path to simulate
        (range is ]0,+inf[, default is 0.5).

    bending_elasticity : float
        Elasticity of the global perturbation bending the fiber path to be
        simulated (range is ]0,+inf[, default is 2).

    bending_force : float
        Force of the global perturbation bending the fiber path to be simulated
        (range is ]0+inf[, default is 0.5).

    Returns
    -------
    numpy.ndarray
        The coordinates of the points as an (2,N) array.

    See Also
    --------
    fiber_spline : Simulate fiber path with a spline.
    fiber_disconnections : Simulate disconnections along fiber path.
    fiber_inhomogeneity : Simulate signal inhomogeneity along fiber path.
    fibers : Simulate a set of fibers with given parameters.
    rfibers : Simulate a set of fibers with random parameters within ranges.
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
               / bending_elasticity**2)
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

    Parameters
    ----------
    num_of_points : strictly positive int
        Number of points on the fiber.

    number_of_channels : int
        Number of channels of the output image.

    pattern : List[int]
        The input patterns of the fibers to simulate.

    length : List[float]
        The lengths of the input pattern segments to simulate.

    local_force : float
        Force of local signal inhomogeneity (default is 0.05).

    global_force : float
        Force of the global signal inhomogeneity (default is 0.25).

    global_rate : float
        Rate of modulation of the global signal inhomogeneity (default is 1.5).

    Returns
    -------
    numpy.ndarray
        Sequence of inhomogeneity factors.

    See Also
    --------
    fiber_spline : Simulate fiber path with a spline.
    fiber_disconnections : Simulate disconnections along fiber path.
    fibers : Simulate a set of fibers with given parameters.
    rfibers : Simulate a set of fibers with random parameters within ranges.
    """
    t = np.arange(0, num_of_points)
    u = t / t.max()
    length_cumsum = np.divide(np.cumsum([0.] + length), sum(length))
    s = np.zeros((number_of_channels, num_of_points))

    # create global inhomogeneity
    global_rate *= num_of_points
    d = np.exp(-0.5 * (t - np.abs(0.5 *
                                  (t.max()-t.min()) * np.random.randn()))**2
               / global_rate**2)

    for i in range(len(s)):
        # create channel signal (following pattern)
        for j in range(len(pattern)):
            if pattern[j] == i:
                s[i, np.bitwise_and(length_cumsum[j] <= u,
                                    u <= length_cumsum[j+1])] = 1
                # local inhomogeneity for current pattern
                s[i] += local_force * np.random.randn(num_of_points)
            else:
                # local inhomogeneity for non-current pattern
                s[i] += np.abs(local_force * np.random.randn(num_of_points))

        # global inhomogeneity
        s[i] *= global_force * d

    return s


def fiber_disconnections(fiber_points, disc_prob=0.2, return_prob=0.5):
    """
    Simulate disconnections in fiber.

    The disconnections are modeled as a state of a Markov process. The points
    sampled on the fiber path define a Markov chain and those points can have
    two states: not disconnected or disconnected. The corresponding
    probabilities rule the random apparitions of the disconnections.

    Parameters
    ----------
    fiber_points : numpy.ndarray
        Points of the input fiber.

    disc_prob : float
        Probability to have disconnections (default is 0.2).

    return_prob : float
        Probability to stop the disconnection (default is 0.5).

    Returns
    -------
    numpy.ndarray
        The coordinates of the degraded points as an (2,N) array.

    See Also
    --------
    fiber_spline : Simulate fiber path with a spline.
    fiber_inhomogeneity : Simulate signal inhomogeneity along fiber path.
    fibers : Simulate a set of fibers with given parameters.
    rfibers : Simulate a set of fibers with random parameters within ranges.
    """
    state = 1  # initial state is no disconnected
    select = []  # points with not disconnected state will be selected

    for _ in fiber_points[0]:
        if state == 1:
            select.append(1-np.random.binomial(1, disc_prob) == 1)
        else:
            select.append(np.random.binomial(1, return_prob) == 1)

    return fiber_points[:, select]


def fiber_paths(geom_props):
    """
    Simulate fiber paths as points in image space (fluophores).

    Parameters
    ----------
    geom_props : List[dict]
        Geometrical properties (see dfa.simulation.fiber_spline).

    Returns
    -------
    List[numpy.ndarray]
        The list of fibers paths (list of coordinates of the points as
        (2,N) arrays).
    """
    paths = []

    for geom_prop in geom_props:
        paths.append(fiber(**geom_prop))

    return paths


def fibers(number_of_channels, patterns, lengths, paths, disc_props,
           signal_props):
    """
    Simulate fibers fluophores as points in image space.

    Parameters
    ----------
    number_of_channels : int
        Number of channels of the output image.

    patterns : List[List[int]]
        The input patterns of the fibers to simulate.

    lengths : List[List[float]]
        The lengths of the input pattern segments to simulate.

    paths : List[numpy.ndarray]
        The list of fibers paths (list of coordinates of the points as
        (2,N) arrays).

    disc_props : List[dict]
        Disconnections properties (see dfa.simulation.fiber_disconnections).

    signal_props : List[dict]
        Signal properties (see dfa.simulation.fiber_inhomogeneity).

    Returns
    -------
    List[(numpy.ndarray, numpy.ndarray)]
        The points/signal of fibers paths simulated.

    See Also
    --------
    fiber_spline : Simulate fiber path with a spline.
    fiber_disconnections : Simulate disconnections along fiber path.
    fiber_inhomogeneity : Simulate signal inhomogeneity along fiber path.
    rfibers : Simulate a set of fibers with random parameters within ranges.
    """
    fiber_objects = []

    for pattern, length, path, disc_prop, signal_prop in \
            zip(patterns, lengths, paths, disc_props, signal_props):
        f = fiber_disconnections(path, **disc_prop)
        s = fiber_inhomogeneity(f.shape[1], number_of_channels,
                                pattern, length, **signal_prop)
        fiber_objects.append((f, s))

    return fiber_objects


def _uniform_sample_within_range(sampling_range, sample_number):
    return np.abs(np.diff(sampling_range)) * \
           np.random.rand(sample_number) + \
           np.min(sampling_range)


def rpaths(number, angle_range, shift_range, perturbations_force_range,
           bending_elasticity_range, bending_force_range,
           model=modeling.standard):
    """
    Randomly simulate fiber paths.

    For information abount the ranges, refer to dfa.simulation.fiber_spline,
    dfa.simulation.fiber_disconnections and
    dfa.simulation.inhomogeneity.

    See Also
    --------
    fiber_spline : Simulate fiber path with a spline.
    fiber_disconnections : Simulate disconnections along fiber path.
    fiber_inhomogeneity : Simulate signal inhomogeneity along fiber path.
    fibers : Simulate a set of fibers with given parameters.
    rfibers : Simulate a set of fibers with random parameters within ranges.
    """
    patterns, lengths = model.simulate_patterns(number)

    geom_props = []

    angles = _uniform_sample_within_range(angle_range, number)
    shifts_x = _uniform_sample_within_range(shift_range[0], number)
    shifts_y = _uniform_sample_within_range(shift_range[1], number)
    perturbations_forces = _uniform_sample_within_range(
        perturbations_force_range, number)
    bending_elasticities = _uniform_sample_within_range(
        bending_elasticity_range, number)
    bending_forces = _uniform_sample_within_range(bending_force_range, number)

    for i in range(number):
        geom_props.append({
            'angle': angles[i], 'length': sum(lengths[i]),
            'shift': (shifts_x[i], shifts_y[i]),
            'perturbations_force': perturbations_forces[i],
            'bending_elasticity': bending_elasticities[i],
            'bending_force': bending_forces[i]})

    return fiber_paths(geom_props), patterns, lengths


def rfibers(number, patterns, lengths, paths, disc_prob_range,
            return_prob_range, local_force_range, global_force_range,
            global_rate_range, model=modeling.standard):
    """
    Randomly simulate fibers objects with geometrical deterioration /
    signal inhomogeneity.

    For information abount the ranges, refer to dfa.simulation.fiber_spline,
    dfa.simulation.fiber_disconnections and
    dfa.simulation.inhomogeneity.

    See Also
    --------
    fiber_spline : Simulate fiber path with a spline.
    fiber_disconnections : Simulate disconnections along fiber path.
    fiber_inhomogeneity : Simulate signal inhomogeneity along fiber path.
    fibers : Simulate a set of fibers with given parameters.
    rfibers : Simulate a set of fibers with random parameters within ranges.
    """
    number_of_channels = len(model.channels_names)

    disc_props = []

    disc_probs = _uniform_sample_within_range(disc_prob_range, number)
    return_probs = _uniform_sample_within_range(return_prob_range, number)

    signal_props = []

    local_forces = _uniform_sample_within_range(local_force_range, number)
    global_forces = _uniform_sample_within_range(global_force_range, number)
    global_rates = _uniform_sample_within_range(global_rate_range, number)

    for i in range(number):
        disc_props.append({
            'disc_prob': disc_probs[i], 'return_prob': return_probs[i]})
        signal_props.append({
            'local_force': local_forces[i], 'global_force': global_forces[i],
            'global_rate': global_rates[i]})

    return fibers(number_of_channels, patterns, lengths,
                  paths, disc_props, signal_props)


def image_by_diffraction(shape, fibers_points, fibers_signal, psf,
                         positions=None):
    """
    Create a diffraction limited image from points along fiber paths.

    Parameters
    ----------
    shape : (int, int)
        Shape of the output image.

    fibers_points : List[numpy.ndarray]
        Coordinates points in image space of the fiber paths.

    fibers_signal : List[numpy.ndarray]
        Signal power of each points along fiber paths.

    psf : numpy.ndarray
        3D image of the PSF used for diffraction simulation.

    positions : list of int or None
        Positions in the PSF z-stack used to simulate out-of-focus. When set to
        None (default), the positions are 0 (in-focus).

    Returns
    -------
    numpy.ndarray
        Simulated diffraction-limited image.

    See Also
    --------
    shot_noise : Simulate photon noise.
    image : Simulate image acquisition conditions of fibers with given SNR,
    z-positions and fiber objects.
    rimage : Simulate image acquisition conditions with given SNR and fiber
    objects and random z-positions.
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

    Parameters
    ----------
    input_image : numpy.ndarray with 2 dimensions
        Input noise-free image.

    snr : float
        Signal-to-noise ratio of the simulated image (in dB).

    Returns
    -------
    numpy.ndarray with same shape as input_image
        Image corrupted with photon-noise (Poisson distribution).

    See Also
    --------
    image_by_diffraction : Simulate diffraction of microscopy lenses.
    image : Simulate image acquisition conditions of fibers with given SNR,
    z-positions and fiber objects.
    rimage : Simulate image acquisition conditions with given SNR and fiber
    objects and random z-positions.
    """
    # When poisson noise, we have parameter lambda = 10^(SNR_dB / 5)
    for i in range(input_image.shape[0]):
        input_image[i] = np.round(input_image[i] * np.power(10, snr / 5))
        background = np.less_equal(input_image[i], 0)
        input_image[i, background] = np.random.rand(
            input_image[i, background].size)
        input_image[i] = np.random.poisson(
            np.round(input_image[i]).astype(int))

    return input_image


def image(fiber_objects, shape, zindices, psf, snr=20):
    """
    Simulate image acquisition conditions of fiber objects.

    The fiber objects need to be first simulated using the appropriate
    functions. With each object is associated a z-index giving the relative
    position of the fiber object plane to the focal plane of the final image.

    Parameters
    ----------
    fiber_objects : List[(numpy.ndarray, numpy.ndarray)] with 2 dimensions
        Input simulated fiber objects and signals.

    shape : (int, int)
        Shape of the output image.

    zindices : list of int (or iterable)
        Plane positions of fibers relative to the focal plane.

    psf : numpy.ndarray with 3 dimensions
        PSF used to simulate the microscope's diffraction of light.

    snr : float
        Signal-to-noise ratio of the simulated image (in dB).

    Returns
    -------
    numpy.ndarray with 2 dimensions and shape outshape
        Final simulated image of fibers with acquisition artefacts.

    See Also
    --------
    image_by_diffraction : Simulate diffraction of microscopy lenses.
    shot_noise : Simulate photon noise.
    rimage : Simulate image acquisition conditions with given SNR and fiber
    objects and random z-positions.
    """
    clean_image = image_by_diffraction(
        shape, *zip(*fiber_objects), psf, zindices)

    return shot_noise(clean_image, snr)


def rimage(fiber_objects, shape, zindex_range, psf, snr=10):
    """
    Simulate image acquisition conditions of fiber objects with random
    out-of-focus effects.

    Parameters
    ----------
    fiber_objects : List[(numpy.ndarray, numpy.ndarray)] with 2 dimensions
        Input simulated fiber objects and signals.

    shape : (int, int)
        Shape of the output image.

    zindex_range : (int, int)
        Plane positions range of fibers relative to the focal plane.

    psf : numpy.ndarray with 3 dimensions
        PSF used to simulate the microscope's diffraction of light.

    snr : float
        Signal-to-noise ratio of the simulated image (in dB).

    Returns
    -------
    numpy.ndarray with 2 dimensions and shape outshape
        Final simulated image of fibers with acquisition artefacts.

    See Also
    --------
    image_by_diffraction : Simulate diffraction of microscopy lenses.
    shot_noise : Simulate photon noise.
    image : Simulate image acquisition conditions of fibers with given SNR,
    z-positions and fiber objects.
    """
    if zindex_range is None:
        zindices = None
    else:
        zindices = np.random.randint(min(zindex_range), max(zindex_range),
                                     size=len(fiber_objects)).tolist()

    return image(fiber_objects, shape, zindices, psf, snr)

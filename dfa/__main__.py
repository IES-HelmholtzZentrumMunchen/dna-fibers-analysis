"""
Main entry point of the dfa package.
This can be used to actual run the analysis (partially of fully).
"""

import numpy as np
from dfa import _utilities as _ut


def pipeline_command(args):
    import os
    import progressbar
    from dfa import detection as det
    from dfa import extraction as ex
    from matplotlib import pyplot as plt

    # non-user parameters
    alpha = 0.5
    smoothing = 20
    min_length = 30

    radius = 5

    # read inputs
    images, names, masks = _ut.read_inputs(args.input, args.masks, '.tif')

    # process image by image
    with progressbar.ProgressBar(max_value=len(images)) as bar:
        for num, (image, name) in enumerate(zip(images, names)):
            # pre-processing
            if len(image.shape) != 3:
                raise ValueError('Input image {} does not have 3 channels! '
                                 'Its shape is {}! '
                                 'Only two-channels images are supported!'
                                 .format(name, len(image.shape)))

            flat_image = image.sum(axis=0)

            # detection
            fibers = det.detect_fibers(
                flat_image,
                scales=[args.fiber_size - 1,
                        args.fiber_size,
                        args.fiber_size + 1],
                alpha=alpha,
                beta=1 / args.intensity_sensitivity,
                length=args.reconstruction_extent,
                size=args.fiber_size,
                smoothing=smoothing,
                min_length=min_length,
                extent_mask=None)

            if args.save_detected_fibers:
                _ut.write_points_to_txt(
                    args.output,
                    os.path.basename(name),
                    fibers)

            # extraction
            extracted_fibers = ex.unfold_fibers(
                image, fibers, radius=radius)

            if args.save_extracted_profiles:
                for number, extracted_fiber, in enumerate(extracted_fibers):
                    np.savetxt(os.path.join(
                        args.output, '{}_fiber{}.csv'.format(
                            name, number + 1)),
                        np.vstack((range(extracted_fiber.shape[2]),
                                   extracted_fiber[0].sum(axis=0),
                                   extracted_fiber[1].sum(axis=0))).T,
                        delimiter=',', header='X, Y1, Y2', comments='')

            if args.save_extracted_fibers:
                figures = _ut.create_figures_from_fibers_images(
                    [name], [extracted_fibers], radius, group_fibers=False)

                for filename, fig in figures:
                    fig.savefig(os.path.join(args.output, filename))

            if args.save_grouped_fibers:
                figures = _ut.create_figures_from_fibers_images(
                    [name], [extracted_fibers], radius, group_fibers=True)

                for filename, fig in figures:
                    fig.savefig(os.path.join(args.output, filename))

            bar.update(num + 1)


def detection_command(args):
    from dfa import detection as det

    images, names, masks = _ut.read_inputs(args.input, args.mask, '.tif')

    for image, name, mask in zip(images, names, masks):
        if len(image.shape) == 2:
            fiber_image = image
        else:
            fiber_image = image.sum(axis=0)

        coordinates = det.detect_fibers(
            fiber_image,
            scales=np.linspace(args.scales[0], args.scales[1],
                               int(args.scales[2])).tolist(),
            alpha=args.fiber_sensitivity,
            beta=1 / args.intensity_sensitivity,
            length=args.reconstruction_extent,
            size=(args.scales[0]+args.scales[1])/2,
            smoothing=args.smoothing,
            min_length=args.fibers_minimal_length,
            extent_mask=mask)

        if args.output is None:
            from matplotlib import pyplot as plt
            plt.imshow(fiber_image, cmap='gray', aspect='equal')
            for c in coordinates:
                plt.plot(*c, '-r')
            plt.show()
        else:
            _ut.write_points_to_txt(
                args.output, name, coordinates)


def extraction_command(args):
    import os
    from matplotlib import pyplot as plt
    from skimage import io
    from dfa import extraction as ex

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
    extracted_fibers = ex.extract_fibers(
        input_images, input_fibers, radius=args.radius)

    # output
    figures = _ut.create_figures_from_fibers_images(
        input_names, extracted_fibers, args.radius, args.group_fibers)

    if args.output is None:
        plt.show()
    else:
        # export to csv the profiles
        for image_extracted_fiber, input_name \
                in zip(extracted_fibers, input_names):
            for number, extracted_fiber, in enumerate(image_extracted_fiber):
                np.savetxt(os.path.join(args.output,
                                        '{}_fiber{}.csv'.format(input_name,
                                                                number + 1)),
                           np.vstack((range(extracted_fiber.shape[2]),
                                      extracted_fiber[0].sum(axis=0),
                                      extracted_fiber[1].sum(axis=0))).T,
                           delimiter=',', header='X, Y1, Y2', comments='')

        # export to png the grouped fibers or the single fibers + profiles
        if figures is not None:
            for name, fig in figures:
                fig.savefig(os.path.join(args.output, name))


def analysis_command(args):
    import os
    import copy
    from dfa import modeling as mod
    from dfa import analysis as ana

    # Check inputs (because argparse cannot manage 2+ nargs
    if len(args.input_columns) < 2:
        parser.error('argument --input_columns: expected at '
                     'least two arguments')

    if len(args.channels_names) < 2:
        parser.error('argument --channels_names: expected at '
                     'least two arguments')

    if len(args.input_columns) != len(args.channels_names):
        parser.error('arguments --input_columns and --channels_names: '
                     'expected the same number of arguments')

    # Read profiles from input path
    if os.path.isfile(args.input):
        if not args.input.endswith('.csv'):
            raise ValueError('The input file must be a csv file!')

        paths = [args.input]
    elif os.path.isdir(args.input):
        if args.recursive:
            paths = [os.path.join(root, filename)
                     for root, _, filenames in os.walk(args.input)
                     for filename in filenames
                     if filename.endswith('.csv')]
        else:
            paths = [os.path.join(args.input, filename)
                     for filename in os.listdir(args.input)
                     if filename.endswith('.csv')]

        if len(paths) == 0:
            raise ValueError('The input folder does not contain any csv file!')
    else:
        raise ValueError('The input is neither a valid file nor '
                         'a valid directory!')

    profiles = [np.loadtxt(path, delimiter=',', skiprows=1,
                           usecols=[0] + args.input_columns)
                for path in paths]

    # Get data origin information (keys)
    if args.keys_in_file is None:
        keys = [tuple(path[:-4].split('/')[-len(args.scheme):])
                for path in paths]
    else:
        keys = [tuple(path.split('/')[-1][:-4]
                      .split(args.keys_in_file)[-len(args.scheme):])
                for path in paths]

    # Quantify
    if args.model is None:
        model = copy.deepcopy(mod.standard)
    else:
        if not os.path.isfile(args.model):
            raise ValueError('The input model argument must be a valid path'
                             ' to a text file!')

        model = mod.Model.load(args.model)

    model.initialize_model()
    detailed_analysis = ana.analyzes(
        profiles, model=model, keys=keys, keys_names=args.scheme,
        discrepancy=args.discrepancy, contrast=args.contrast,
        channels_names=args.channels_names)

    # Display or save results
    if args.output is None:
        print(detailed_analysis)
    else:
        detailed_analysis.to_csv(args.output)

    if args.output_model is None:
        model.print()
    else:
        model.save(args.output_model)


def simulate_command(args):
    import os
    from skimage import io
    from dfa import modeling as mod
    from dfa import simulation as sim
    from dfa import _utilities as _ut

    if args.model is None:
        args.model = mod.standard
    else:
        args.model = mod.Model.load(args.model)

    if args.output is not None and \
            not os.path.exists(os.path.dirname(args.output)):
        raise ValueError('The output path does not exist!')

    simulated_psf = io.imread(args.psf_file)

    fibers_objects = sim.rfibers(
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

    degraded_image = sim.rimage(
        fiber_objects=fibers_objects, shape=args.shape,
        zindex_range=args.z_index, psf=simulated_psf, snr=args.snr)

    if args.output is None:
        from matplotlib import pyplot as plt

        display_image = np.zeros(degraded_image.shape[1:] + (3,), dtype='uint8')
        display_image[:, :, 0] = 255 * \
            _ut.norm_min_max(degraded_image[0], degraded_image)
        display_image[:, :, 1] = 255 * \
            _ut.norm_min_max(degraded_image[1], degraded_image)

        plt.imshow(display_image, aspect='equal')

        plt.show()
    else:
        path = os.path.dirname(args.output)
        name, _ = os.path.splitext(os.path.basename(args.output))

        io.imsave(args.output, degraded_image.astype('int16'))

        import _utilities as _ut

        fibers_output = os.path.join(path, '{}_fibers'.format(name))
        os.mkdir(fibers_output)
        _ut.write_points_to_txt(
            fibers_output, name,
            [fiber_object for fiber_object, _ in fibers_objects])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='python3 -m dfa',
        description='DNA fibers analysis pipeline in python.')
    subparsers = parser.add_subparsers(
        title='available commands', dest='command')
    subparsers.required = True

    # pipeline command
    parser_pipeline = subparsers.add_parser(
        'pipeline', help='run the full analysis pipeline',
        description='Run the full analysis pipeline, though with a limited '
                    'set of accessible parameters.')
    parser_pipeline.set_defaults(func=pipeline_command)

    pipeline_inout = parser_pipeline.add_argument_group('Input/Output')
    pipeline_inout.add_argument(
        'input', type=_ut.check_valid_path,
        help='Path to input image(s). It can be either folder or file.')
    pipeline_inout.add_argument(
        'output', type=_ut.check_valid_directory,
        help='Path to output directory, where all the outputs will be saved.')
    pipeline_inout.add_argument(
        '--masks', type=_ut.check_valid_or_empty_path, default='',
        help='Path to input masks of images (default is automatic masking).')
    pipeline_inout.add_argument(
        '--save-detected-fibers', action='store_true',
        help='Save intermediate files of detected fibers (default is not '
             'saving).')
    pipeline_inout.add_argument(
        '--save-extracted-fibers', action='store_true',
        help='Save intermediate files of extracted fibers (default is not '
             'saving).')
    pipeline_inout.add_argument(
        '--save-grouped-fibers', action='store_true',
        help='Save intermediate files of extracted fibers grouped by image '
             '(default is not saving).')
    pipeline_inout.add_argument(
        '--save-extracted-profiles', action='store_true',
        help='Save intermediate files of extracted profiles (default is not '
             'saving).')

    pipeline_detection = parser_pipeline.add_argument_group('Detection')
    pipeline_detection.add_argument(
        '--intensity-sensitivity',
        type=_ut.check_positive_float, default=0.7,
        help='Sensitivity of detection to intensity in percentage (default is '
             '0.5, valid range is ]0, +inf[).')
    pipeline_detection.add_argument(
        '--fiber-size', type=float, default=3,
        help='Size in pixels of fiber''s average width (default is 3).')
    pipeline_detection.add_argument(
        '--reconstruction-extent',
        type=_ut.check_positive_int, default=20,
        help='Reconstruction extent in pixels (default is 20, range is '
             ']0, +inf[).')

    # detection command
    parser_detection = subparsers.add_parser(
        'detect', help='fibers detection in images',
        description='Automatically detect DNA fibers in single '
                    'confocal sections.')
    parser_detection.set_defaults(func=detection_command)

    detection_images = parser_detection.add_argument_group('Images')
    detection_images.add_argument(
        'input', type=_ut.check_valid_path, help='Path to input image.')

    detection_detection = parser_detection.add_argument_group('Detection')
    detection_detection.add_argument(
        '--fiber-sensitivity',
        type=_ut.check_float_0_1, default=0.5,
        help='Sensitivity of detection to geometry in percentage (default is '
             '0.5, valid range is ]0, 1]).')
    detection_detection.add_argument(
        '--intensity-sensitivity',
        type=_ut.check_positive_float, default=0.7,
        help='Sensitivity of detection to intensity in percentage (default is '
             '0.5, valid range is ]0, +inf[).')
    detection_detection.add_argument(
        '--scales', type=_ut.check_scales, nargs=3, default=[2, 4, 3],
        help='Scales to use in pixels (minimum, maximum, number of scales). '
             'Default is 2 4 3.')
    detection_detection.add_argument(
        '--mask', type=_ut.check_valid_or_empty_path, default='',
        help='Mask where to search for fibers (default is automatic masking).')

    detection_reconstruction = \
        parser_detection.add_argument_group('Reconstruction')
    detection_reconstruction.add_argument(
        '--no-flat', action='store_true',
        help='Use non-flat structuring elements for fiber reconstruction (by '
             'default use flat structuring elements).')
    detection_reconstruction.add_argument(
        '--reconstruction-extent',
        type=_ut.check_positive_int, default=20,
        help='Reconstruction extent in pixels (default is 20, range is '
             ']0, +inf[).')

    detection_medial = parser_detection.add_argument_group('Medial axis')
    detection_medial.add_argument(
        '--smoothing', type=_ut.check_positive_int, default=20,
        help='Smoothing of the output fibers (default is 20, range is is '
             ']0, +inf[).')
    detection_medial.add_argument(
        '--fibers-minimal-length', type=_ut.check_positive_int, default=30,
        help='Minimal length of a fiber in pixels default is 30, range is '
             ']0, +inf[).')
    detection_medial.add_argument(
        '--output', type=_ut.check_valid_path, default=None,
        help='Output path for saving detected fibers (default is None).')

    # extraction command
    parser_extraction = subparsers.add_parser(
        'extract',
        help='extract detected fibers from images',
        description='Extract and unfold detected fibers and their associated '
                    'profiles.')
    parser_extraction.set_defaults(func=extraction_command)

    parser_extraction.add_argument(
        'input', type=_ut.check_valid_path,
        help='Path to single image file or path to folder with multiple '
             'images.')
    parser_extraction.add_argument(
        'fibers', type=_ut.check_valid_directory,
        help='Path to directory containing fiber files.')
    parser_extraction.add_argument(
        '--radius', type=_ut.check_positive_int, default=5,
        help='Radius of the fibers to extract from images (default is 5).')
    parser_extraction.add_argument(
        '--group-fibers', action='store_true',
        help='Group the extracted fibers by image.')
    parser_extraction.add_argument(
        '--output', type=_ut.check_valid_path, default=None,
        help='Output path for saving profiles and extracted fibers (default '
             'is None).')

    # analysis command
    parser_analysis = subparsers.add_parser(
        'analyze',
        help='analyze the extracted fibers',
        description='Automatically assign DNA fiber model pattern to fibers '
                    'and quantify lengths and rates.')
    parser_analysis.set_defaults(func=analysis_command)

    group_profile = parser_analysis.add_argument_group('Profiles')
    group_profile.add_argument(
        'input', type=str,
        help='Input path to profile(s) (folder or file). Profiles are assumed '
             'to have at least 3 columns, the first one being the values of '
             'the x-axis.')
    group_profile.add_argument(
        '--channels_names', type=str, nargs='+', default=['CIdU', 'IdU'],
        help='Names of the channels as they appear in order in the profiles '
             '(default is CIdU and IdU).')
    group_profile.add_argument(
        '--input_columns', type=int, nargs='+', default=[1, 2],
        help='Columns index of the profiles to use')
    group_profile.add_argument(
        '--recursive', action='store_true',
        help='Search in specified path recursively (default is False; works '
             'only for directory input).')

    group_model = parser_analysis.add_argument_group('Model')
    group_model.add_argument(
        '--model', type=str, default=None,
        help='Path to the model to use (default will use the standard model '
             'defined in the dfa.modeling module).')
    group_model.add_argument(
        '--discrepancy', type=float, default=0,
        help='Discrepancy regularization on intensity of branches of the same '
             'channel (default is 0, i.e. deactivated).')
    group_model.add_argument(
        '--contrast', type=float, default=0,
        help='Contrast regularization between intensities of branches of '
             'opposite channels (default is 0, i.e. deactivated).')
    group_model.add_argument(
        '--output_model', type=str, default=None,
        help='Output path for saving the model (default is None).')

    group_data = parser_analysis.add_argument_group('Quantification')
    group_data.add_argument(
        '--scheme', type=str, nargs='+',
        default=['experiment', 'image', 'fiber'],
        help='Names of the keys used as indexing of the results (default is '
             'experiment, image, fiber; there should be at least one name).')
    group_data.add_argument(
        '--keys_in_file', type=str, default=None,
        help='If set, the keys are searched in the filenames (separator must '
             'be provided); otherwise the keys are searched in the last path '
             'elements (folders and filenames separated by /).')
    group_data.add_argument(
        '--output', type=str, default=None,
        help='Output path for saving data analysis (default is None).')

    # simulation command
    parser_simulation = subparsers.add_parser(
        'simulate',
        help='simulate an image containing fibers',
        description='Simulate a single confocal section containing DNA fibers.')
    parser_simulation.set_defaults(func=simulate_command)

    parser_simulation.add_argument(
        '--output', type=_ut.check_valid_path, default=None,
        help='Output path for saving simulation (default: None).')

    fibers_group = parser_simulation.add_argument_group('Fibers')
    fibers_group.add_argument(
        '--number', type=int, default=30,
        help='Number of fiber segments to simulate (default: 30).')
    fibers_group.add_argument(
        '--orientation', type=float, nargs=2, default=[30, 40],
        help='Orientation range of fibers (in degrees, default: [30, 40]).')
    fibers_group.add_argument(
        '--model', type=str, default=None,
        help='Path to model file (default: internal standard).')
    fibers_group.add_argument(
        '--location', type=float, nargs=4, default=[100, 924, 100, 924],
        help='Coordinates range of fiber center (<x_min> <x_max> <y_min> '
             '<y_max>, in pixels, default: [100, 924, 100, 924]).')
    fibers_group.add_argument(
        '--perturbations-force-range', type=float, nargs=2, default=[0.1, 0.3],
        help='Local perturbations of the fiber path (default is [0.1, 0.2]).')
    fibers_group.add_argument(
        '--bending-elasticity-range', type=float, nargs=2, default=[1, 3],
        help='Elasticity of the global bending of the fiber path (default is '
             '[1, 3]).')
    fibers_group.add_argument(
        '--bending-force-range', type=float, nargs=2, default=[10, 20],
        help='Force of the global bending of the fiber path (default is '
             '[10, 20]).')
    fibers_group.add_argument(
        '--disconnection-probability-range', type=float, nargs=2,
        default=[0.05, 0.1],
        help='Probability of disconnections happening on the fiber path '
             '(default is [0.05, 0.1]).')
    fibers_group.add_argument(
        '--return-probability-range', type=float, nargs=2, default=[0.5, 0.7],
        help='Probability of recovering when the fiber path is disconnected '
             '(default is [0.5, 0.7]).')
    fibers_group.add_argument(
        '--local-force-range', type=float, nargs=2, default=[0.05, 0.2],
        help='Force of local signal inhomogeneity (default is [0.05, 0.2]).')
    fibers_group.add_argument(
        '--global-force-range', type=float, nargs=2, default=[2, 5],
        help='Force of global signal inhomogeneity (default is [2, 5]).')
    fibers_group.add_argument(
        '--global-rate-range', type=float, nargs=2, default=[0.5, 1.5],
        help='Rate of global signal inhomogeneity (default is [0.5, 1.5]).')

    image_group = parser_simulation.add_argument_group('Image degradations')
    image_group.add_argument(
        '--shape', type=int, nargs=2, default=[1024, 1024],
        help='Shape of the output image (default is [1024, 1024]). Note that '
             'the pixel size is the same as the PSF pixel size.')
    image_group.add_argument(
        'psf_file', type=str, default=None, help='Path to 3D PSF file.')
    image_group.add_argument(
        '--z_index', type=int, nargs=2, default=[-1, 1],
        help='Z-index of fiber objects (default: [-1, 1]).')
    image_group.add_argument(
        '--snr', type=float, default=7, help='SNR in decibels (default: 7).')

    # parsing
    args = parser.parse_args()
    args.func(args)

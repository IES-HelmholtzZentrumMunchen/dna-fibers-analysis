"""
Main entry point of the dfa package.
This can be used to actual run the analysis (partially of fully).
"""

import warnings

import numpy as np

from dfa import utilities as ut


def pipeline_command(args):
    """Run the full DFA pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
    import os
    import copy
    import progressbar
    import pandas as pd
    from matplotlib import pyplot as plt
    from dfa import detection as det
    from dfa import extraction as ex
    from dfa import modeling as mod
    from dfa import analysis as ana

    progressbar.streams.wrap_stderr()

    def _create_if_not_existing(output, name):
        save_path = os.path.join(output, name)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        return save_path

    # non-user parameters
    alpha = 0.5
    smoothing = 20
    min_length = 30

    radius = 5

    scheme = args.scheme + ['fiber']

    # read inputs
    images, names, masks = ut.read_inputs(args.input, args.masks, '.tif')

    # initialize output
    if args.model is None:
        model = copy.deepcopy(mod.standard)
    else:
        if not os.path.isfile(args.model):
            raise ValueError(
                'The input model argument must be a valid path'
                ' to a text file!')

        model = mod.Model.load(args.model)

    model.initialize_model()

    detailed_analysis = pd.DataFrame(
        [], columns=['pattern', 'channel', 'length'],
        index=pd.MultiIndex(levels=[[] for _ in range(len(scheme))],
                            labels=[[] for _ in range(len(scheme))],
                            name=scheme))

    # process image by image
    with progressbar.ProgressBar(max_value=len(images)) as bar:
        for num, (image, name, mask) in enumerate(zip(images, names, masks)):
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
                user_mask=mask)

            if args.save_all or args.save_detected_fibers:
                ut.write_fibers(fibers,
                                _create_if_not_existing(args.output, 'fibers'),
                                os.path.basename(name))

                plt.imshow(flat_image, cmap='gray', aspect='equal')
                indices = []
                for k, c in enumerate(fibers):
                    plt.plot(*c, '-c')
                    plt.text(*c.mean(axis=1), str(k + 1), color='c')
                    indices.append(k + 1)
                plt.title('Fibers of {}'.format(name))
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(
                    _create_if_not_existing(args.output, 'fibers'),
                    '{}_fibers.pdf'.format(name)))
                plt.close()

            # extraction
            # the coordinates of the fibers are sorted such that the profiles
            # are extracted in the same orientation for any input.
            extracted_fibers = ex.unfold_fibers(
                image, [fiber[:, np.lexsort((*fiber,))]
                        for fiber in ut.resample_fibers(fibers, 1)],
                radius=radius)

            extracted_profiles = [
                ex.extract_profiles_from_fiber(extracted_fiber,
                                               pixel_size=args.pixel_size)
                for extracted_fiber in extracted_fibers]

            if args.save_all or args.save_extracted_profiles:
                ut.write_profiles(_create_if_not_existing(args.output,
                                                          'profiles'),
                                  name, extracted_profiles)

            if args.save_all or args.save_grouped_fibers:
                figures = ut.create_figures_from_fibers_images(
                    [name], [extracted_fibers], radius, group_fibers=True)

                for filename, fig in figures:
                    fig.savefig(os.path.join(
                        _create_if_not_existing(args.output, 'profiles'),
                        filename))
                    plt.close(fig)

            # analysis
            keys = [tuple(name.split('-')[-len(args.scheme):]) + (num + 1,)
                    for num in range(len(extracted_profiles))]

            current_analysis = ana.analyzes(
                extracted_profiles, model=model, update_model=False,
                keys=keys, keys_names=scheme,
                discrepancy=args.discrepancy, contrast=args.contrast)

            if args.save_all or args.save_extracted_fibers:
                figures = ut.create_figures_from_fibers_images(
                    [name], [extracted_fibers], radius, group_fibers=False,
                    analysis=current_analysis)

                for filename, fig in figures:
                    fig.savefig(os.path.join(
                        _create_if_not_existing(args.output, 'profiles'),
                        filename))
                    plt.close(fig)

            detailed_analysis = detailed_analysis.append(current_analysis)

            bar.update(num + 1)

    detailed_analysis.to_csv(
        os.path.join(_create_if_not_existing(args.output, 'analysis'),
                     args.output_name + '.csv'))

    if args.save_all or args.save_model:
        model.update_model()
        model.save(
            os.path.join(_create_if_not_existing(args.output, 'analysis'),
                         args.output_name + '_model.txt'))


def detection_command(args):
    """Run the fiber detection process.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
    from os import path as op
    import progressbar
    from matplotlib import pyplot as plt
    from dfa import detection as det

    images, names, masks = ut.read_inputs(args.input, args.mask, '.tif')

    with progressbar.ProgressBar(max_value=len(images)) as bar:
        for num, (image, name, mask) in enumerate(zip(images, names, masks)):
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
                user_mask=mask)

            plt.imshow(fiber_image, cmap='gray', aspect='equal')
            indices = []
            for k, c in enumerate(coordinates):
                plt.plot(*c, '-c')
                plt.text(*c.mean(axis=1), str(k + 1), color='c')
                indices.append(k + 1)
            plt.title('Fibers of {}'.format(name))
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            if args.output is None:
                plt.show()
            else:
                plt.savefig(op.join(args.output, '{}_fibers.pdf'.format(name)))
                ut.write_fibers(coordinates, args.output, name, indices=indices)

            bar.update(num + 1)


def extraction_command(args):
    """Run the fiber extraction process.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
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
    input_fibers_indices = []
    input_names = []

    for filename in image_files:
        basename, ext = os.path.splitext(filename)

        if ext == '.tif':
            input_images.append(
                io.imread(os.path.join(image_path, filename)))
            current_fibers = list(zip(*ut.read_fibers(args.fibers,
                                                      image_name=basename)))

            if len(current_fibers) > 0:
                # the coordinates of the fibers are sorted such that the
                # profiles are extracted in the same orientation for any input.
                input_fibers.append(
                    [fiber[:, np.lexsort((*fiber,))]
                     for fiber in ut.resample_fibers(list(current_fibers[0]),
                                                     rate=1)])

                input_fibers_indices.append(list(current_fibers[2]))
                input_names.append(basename)

    # process
    extracted_fibers = ex.extract_fibers(
        input_images, input_fibers, radius=args.radius)

    # output
    if args.profiles_only:
        figures = None
    else:
        figures = ut.create_figures_from_fibers_images(
            input_names, extracted_fibers, args.radius, args.group_fibers,
            input_fibers_indices)

    if args.output is None:
        plt.show()
    else:
        # export to csv the profiles
        for image_extracted_fiber, input_name, input_fibers_index \
                in zip(extracted_fibers, input_names, input_fibers_indices):
            profiles = [ex.extract_profiles_from_fiber(
                extracted_fiber, pixel_size=args.pixel_size)
                        for extracted_fiber in image_extracted_fiber]
            ut.write_profiles(args.output, input_name, profiles,
                              input_fibers_index)

        # export to png the grouped fibers or the single fibers + profiles
        if figures is not None:
            for name, fig in figures:
                fig.savefig(os.path.join(args.output, name))


def analysis_command(args):
    """Run the profiles analysis process.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
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
    keys = []

    for path in paths:
        image, fiber_index = tuple(
            os.path.basename(os.path.splitext(path)[0]).split(
                ut.fiber_indicator))
        keys.append(tuple(image.split('-') + [fiber_index]))

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


def quantification_command(args):
    """
    Run the detailed analysis quantification.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
    import pandas as pd
    from dfa import analysis as ana

    analysis = pd.read_csv(args.input, index_col=args.scheme)

    fork_rate = ana.fork_rate(analysis)
    fork_speed = ana.fork_speed(analysis)
    patterns = ana.get_patterns(analysis)

    if args.output is None:
        print(fork_rate)
        print(fork_speed)
        print(patterns)
    else:
        fork_rate.to_csv('{}_fork_rate.csv'.format(args.output), header=True)
        fork_speed.to_csv('{}_fork_speed.csv'.format(args.output), header=True)
        patterns.to_csv('{}_patterns.csv'.format(args.output), header=True)


def simulate_command(args):
    """Run the fibers image simulation process.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
    import os
    import pickle
    from skimage import io
    from dfa import modeling as mod
    from dfa import simulation as sim
    from dfa import utilities as _ut
    from matplotlib import pyplot as plt

    if args.model is None:
        args.model = mod.standard
    else:
        args.model = mod.Model.load(args.model)

    if args.output is not None and \
            not os.path.exists(os.path.dirname(args.output)):
        raise ValueError('The output path does not exist!')

    simulated_psf = io.imread(args.psf_file)

    patterns = None
    lengths = None

    if args.simulated_fibers == '':
        nums = list(range(1, args.number + 1))
        paths, patterns, lengths = sim.rpaths(
            number=args.number, angle_range=args.orientation,
            shift_range=[tuple(args.location[:2]), tuple(args.location[2:])],
            perturbations_force_range=args.perturbations_force_range,
            bending_elasticity_range=args.bending_elasticity_range,
            bending_force_range=args.bending_force_range)

        if not args.paths_only:
            fibers_objects = sim.rfibers(
                number=args.number,
                patterns=patterns, lengths=lengths, paths=paths,
                disc_prob_range=args.disconnection_probability_range,
                return_prob_range=args.return_probability_range,
                local_force_range=args.local_force_range,
                global_force_range=args.global_force_range,
                global_rate_range=args.global_rate_range, model=args.model)
        else:
            fibers_objects = list(zip(paths, [np.ones(path.shape)
                                              for path in paths]))
    else:
        fibers, _, nums = tuple(zip(*ut.read_fibers(args.simulated_fibers)))
        dirname, basename = os.path.split(args.simulated_fibers)
        basename = os.path.splitext(basename)[0]

        if not args.paths_only:
            signals = list(list(zip(*ut.read_fibers(
                os.path.join(dirname, '{}_signal.zip'.format(basename)))))[0])
            fibers_objects = list(zip(fibers, signals))
        else:
            with open(os.path.join(dirname,
                                   '{}_patterns.pickle'.format(basename)),
                      'rb') as f:
                patterns = pickle.load(f)

            with open(os.path.join(dirname,
                                   '{}_lengths.pickle'.format(basename)),
                      'rb') as f:
                lengths = pickle.load(f)

            # reorder patterns and lengths with fibers numbers
            patterns = [patterns[i-1] for i in nums]
            lengths = [lengths[i-1] for i in nums]

            fibers_objects = sim.rfibers(
                number=args.number,
                patterns=patterns, lengths=lengths, paths=fibers,
                disc_prob_range=args.disconnection_probability_range,
                return_prob_range=args.return_probability_range,
                local_force_range=args.local_force_range,
                global_force_range=args.global_force_range,
                global_rate_range=args.global_rate_range, model=args.model)

    if args.no_image:
        degraded_image = None
    else:
        degraded_image = sim.rimage(
            fiber_objects=fibers_objects, shape=args.shape,
            zindex_range=args.z_index, psf=simulated_psf, snr=args.snr)

    if args.output is None:
        from matplotlib import pyplot as plt

        if degraded_image is not None:
            display_image = np.zeros(degraded_image.shape[1:] + (3,),
                                     dtype='uint8')
            display_image[:, :, 0] = 255 * \
                _ut.norm_min_max(degraded_image[0], degraded_image)
            display_image[:, :, 1] = 255 * \
                _ut.norm_min_max(degraded_image[1], degraded_image)

            plt.imshow(display_image, aspect='equal')

            plt.show()
        else:
            for fiber_object, signal in fibers_objects:
                plt.scatter(*fiber_object, 1, signal[0], cmap='gray')
            plt.show()
    else:
        path = os.path.dirname(args.output)
        name, _ = os.path.splitext(os.path.basename(args.output))

        if degraded_image is not None:
            io.imsave(args.output, degraded_image.astype('int16'))
        else:
            if args.paths_only:
                with open(os.path.join(path, '{}_patterns.pickle'.format(name)),
                          'wb') as f:
                    pickle.dump(patterns, f)

                with open(os.path.join(path, '{}_lengths.pickle'.format(name)),
                          'wb') as f:
                    pickle.dump(lengths, f)

                for fiber_object, fiber_num in zip(fibers_objects, nums):
                    plt.plot(fiber_object[0][0], -fiber_object[0][1], '-k')
                    plt.text(fiber_object[0][0].mean(),
                             -fiber_object[0][1].mean(),
                             str(fiber_num), color='k')
                plt.axes().set_aspect('equal')
                plt.title('Fibers of {}'.format(name))
                plt.xlim(0, 1024)
                plt.ylim(-1024, 0)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(path, '{}_fibers.pdf'.format(name)))
                plt.close()

            ut.write_fibers([signal for _, signal in fibers_objects],
                            path, '{}_signal'.format(name),
                            indices=nums, zipped=True)

        ut.write_fibers([fiber_object for fiber_object, _ in fibers_objects],
                        path, name, indices=nums, zipped=True)


def compare_fibers_command(args):
    """
    Run the detection results comparison process.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
    import pandas as pd
    from dfa import compare as cmp

    def _indices_with_name(names, name):
        return np.indices(names.shape).flatten()[names == name]

    # create and initialize the output data frame
    labels_detection = ['TP', 'FN', 'FP']
    output_precision = pd.DataFrame(
        columns=labels_detection,
        index=pd.MultiIndex(levels=[[] for _ in range(len(args.scheme) - 1)],
                            labels=[[] for _ in range(len(args.scheme) - 1)],
                            names=args.scheme[:-1]))

    labels_accuracy = ['mean dist.', 'median dist.', 'Hausdorff dist.']
    scheme = args.scheme
    scheme.insert(-1, 'expected fiber')
    scheme[-1] = 'actual fiber'
    output_accuracy = pd.DataFrame(
        columns=labels_accuracy,
        index=pd.MultiIndex(levels=[[] for _ in range(len(scheme))],
                            labels=[[] for _ in range(len(scheme))],
                            names=scheme))

    # read the fibers in batch
    expected_fibers = np.array(ut.read_fibers(args.expected)).T
    actual_fibers = np.array(ut.read_fibers(args.actual)).T

    if actual_fibers.size > 0 and expected_fibers.size > 0:
        # to be comparable, we resample the fiber path to a
        # sample every two pixels
        expected_fibers[0] = ut.resample_fibers(expected_fibers[0], rate=2)
        actual_fibers[0] = ut.resample_fibers(actual_fibers[0], rate=2)

        # only fibers with matching image name will be compared;
        # the unique images names are visited sequentially
        unique_image_names = np.unique(
            np.concatenate((expected_fibers[1], actual_fibers[1])))

        for image_name in unique_image_names:
            image_info = image_name.split('-', maxsplit=len(scheme)-3)

            expected_with_name = _indices_with_name(expected_fibers[1],
                                                    image_name)
            actual_with_name = _indices_with_name(actual_fibers[1],
                                                  image_name)

            expected_fibers_with_name = expected_fibers[0][expected_with_name]
            actual_fibers_with_name = actual_fibers[0][actual_with_name]

            matched_fibers = cmp.match_fibers_pairs(
                expected_fibers_with_name, actual_fibers_with_name)

            number_of_matches = 0

            # each pair of matching fibers is compared and distances
            # are appended to the output data frame
            for expected_fiber_index, actual_fiber_index in matched_fibers:
                number_of_matches += 1
                distances = \
                    cmp.fibers_spatial_distances(
                        expected_fibers_with_name[expected_fiber_index],
                        actual_fibers_with_name[actual_fiber_index])

                output_accuracy = output_accuracy.append(
                    pd.Series(
                        dict(zip(labels_accuracy, distances)),
                        name=(*image_info,
                              expected_fibers[2][expected_with_name][
                                  expected_fiber_index],
                              actual_fibers[2][actual_with_name][
                                  actual_fiber_index])))

            output_precision = output_precision.append(
                pd.Series(
                    dict(zip(labels_detection,
                             [number_of_matches,
                              len(expected_fibers_with_name)-number_of_matches,
                              len(actual_fibers_with_name)-number_of_matches])),
                    name=(image_info[0], image_info[1])))

    if args.output is not None:
        output_accuracy.to_csv('{}_accuracy.csv'.format(args.output))
        output_precision.to_csv('{}_precision.csv'.format(args.output))
    else:
        print(output_precision)
        print(output_accuracy)


def comparison_analyses_command(args):
    """
    Run the analysis results comparison process.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
    import pandas as pd
    from dfa import compare as cmp

    expected_analysis = pd.read_csv(args.expected, index_col=args.scheme)
    actual_analysis = pd.read_csv(args.actual, index_col=args.scheme)

    if len(expected_analysis) > 0 and len(actual_analysis) > 0:
        match_scheme = args.scheme.copy()
        match_scheme.insert(-1, 'expected fiber')
        match_scheme[-1] = 'actual fiber'
        fibers_match = pd.read_csv(args.match, index_col=match_scheme)

        pct_match_fibers, match_fibers_expected, match_fibers_actual = \
            cmp.match_index_pairs(expected_analysis, actual_analysis,
                                  fibers_match.index)

        pct_match_patterns, match_patterns_expected, match_patterns_actual = \
            cmp.match_column(match_fibers_expected, match_fibers_actual,
                             column='pattern')

        length_difference = cmp.difference_in_column(
            expected_analysis.ix[match_patterns_expected.index],
            actual_analysis.ix[match_patterns_actual.index],
            column='length')

        new_index = np.array(
            [list(ix_expected) + [ix_actual[-1]]
             for ix_expected, ix_actual in
             zip(expected_analysis.ix[match_patterns_expected.index].index,
                 actual_analysis.ix[match_patterns_actual.index].index)]).T

        match_patterns = pd.DataFrame(
            match_patterns_expected.reset_index()[match_scheme[:2] +
                                                  ['pattern']])
        match_patterns['expected fiber'] = \
            match_patterns_expected.reset_index()['fiber']
        match_patterns['actual fiber'] = \
            match_patterns_actual.reset_index()['fiber']
        match_patterns = match_patterns[match_scheme + ['pattern']]
        match_patterns.set_index(match_scheme, inplace=True)

        length_difference = length_difference.to_frame()

        for i, scheme in enumerate(match_scheme):
            length_difference[scheme] = new_index[i]

        length_difference.set_index(match_scheme, inplace=True)

        if args.output is not None:
            match_patterns.to_csv('{}_patterns.csv'.format(args.output))
            length_difference.to_csv('{}_lengths.csv'.format(args.output))
        else:
            print('percentage of fiber match: {}'.format(
                pct_match_fibers * 100))
            print('percentage of pattern match: {}'.format(
                pct_match_patterns * 100))
            print(match_patterns)
            print(length_difference)
    else:
        print('At least one analysis is empty!')


def create_dataset(args):
    """
    Run the dataset creation process.

    Parameters
    ----------
    args : argparse.Namespace
        Input namespace containing command line arguments.
    """
    from dfa import dataset as dat
    dat.Dataset.create(args.summary, args.images, args.fibers,
                       args.profiles, args.output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='python3 -m dfa',
        description='DNA fibers analysis pipeline in python.')
    subparsers = parser.add_subparsers(
        title='available commands', dest='command')
    subparsers.required = True
    parser.add_argument('--batch', action='store_true')

    # pipeline command
    parser_pipeline = subparsers.add_parser(
        'pipeline', help='run the full analysis pipeline',
        description='Run the full analysis pipeline, though with a limited '
                    'set of accessible parameters.')
    parser_pipeline.set_defaults(func=pipeline_command)

    pipeline_inout = parser_pipeline.add_argument_group('Input/Output')
    pipeline_inout.add_argument(
        'input', type=ut.check_valid_path,
        help='Path to input image(s). It can be either folder or file.')
    pipeline_inout.add_argument(
        'output', type=ut.check_valid_directory,
        help='Path to output directory, where all the outputs will be saved.')
    pipeline_inout.add_argument(
        '--masks', type=ut.check_valid_or_empty_path, default='',
        help='Path to input masks of images (default is automatic masking).')
    pipeline_inout.add_argument(
        '--save-all', action='store_true',
        help='Save all intermediate files (override any other flag).')
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
    pipeline_inout.add_argument(
        '--save-model', action='store_true',
        help='Save output model (default is not saving).')
    pipeline_inout.add_argument(
        '--output-name', type=str, default='detailed_analysis',
        help='Name of the output file containing the analysis (default '
             'is ''detailed_analysis''.')

    pipeline_detection = parser_pipeline.add_argument_group('Detection')
    pipeline_detection.add_argument(
        '--intensity-sensitivity',
        type=ut.check_positive_float, default=0.75,
        help='Sensitivity of detection to intensity in percentage (default is '
             '0.75, valid range is ]0, +inf[).')
    pipeline_detection.add_argument(
        '--fiber-size', type=float, default=3,
        help='Size in pixels of fiber''s average width (default is 3).')
    pipeline_detection.add_argument(
        '--reconstruction-extent',
        type=ut.check_positive_int, default=20,
        help='Reconstruction extent in pixels (default is 20, range is '
             ']0, +inf[).')

    pipeline_analysis = parser_pipeline.add_argument_group('Analysis')
    pipeline_analysis.add_argument(
        '--model', type=ut.check_valid_path, default=None,
        help='Path to the model to use (default will use the standard model '
             'defined in the dfa.modeling module).')
    pipeline_analysis.add_argument(
        '--discrepancy', type=float, default=0,
        help='Discrepancy regularization on intensity of branches of the same '
             'channel (default is 0, i.e. deactivated).')
    pipeline_analysis.add_argument(
        '--contrast', type=float, default=0,
        help='Contrast regularization between intensities of branches of '
             'opposite channels (default is 0, i.e. deactivated).')
    pipeline_analysis.add_argument(
        '--scheme', type=str, nargs='+',
        default=['experiment', 'image'],
        help='Names of the keys used as indexing of the results that can be '
             'found in filename, separated by ''-'' (default is experiment, '
             'image; there should be at least one name).')
    pipeline_analysis.add_argument(
        '--pixel-size', type=float, default=1,
        help='Set the pixel size to the given size in any unit. By default the '
             'pixel size is 1 (no calibration).')

    # detection command
    parser_detection = subparsers.add_parser(
        'detect', help='fibers detection in images',
        description='Automatically detect DNA fibers in single '
                    'confocal sections.')
    parser_detection.set_defaults(func=detection_command)

    detection_images = parser_detection.add_argument_group('Images')
    detection_images.add_argument(
        'input', type=ut.check_valid_path, help='Path to input image.')

    detection_detection = parser_detection.add_argument_group('Detection')
    detection_detection.add_argument(
        '--fiber-sensitivity',
        type=ut.check_float_0_1, default=0.5,
        help='Sensitivity of detection to geometry in percentage (default is '
             '0.5, valid range is ]0, 1]).')
    detection_detection.add_argument(
        '--intensity-sensitivity',
        type=ut.check_positive_float, default=0.75,
        help='Sensitivity of detection to intensity in percentage (default is '
             '0.75, valid range is ]0, +inf[).')
    detection_detection.add_argument(
        '--scales', type=ut.check_scales, nargs=3, default=[2, 4, 3],
        help='Scales to use in pixels (minimum, maximum, number of scales). '
             'Default is 2 4 3.')
    detection_detection.add_argument(
        '--mask', type=ut.check_valid_or_empty_path, default='',
        help='Mask where to search for fibers (default is automatic masking).')

    detection_reconstruction = \
        parser_detection.add_argument_group('Reconstruction')
    detection_reconstruction.add_argument(
        '--no-flat', action='store_true',
        help='Use non-flat structuring elements for fiber reconstruction (by '
             'default use flat structuring elements).')
    detection_reconstruction.add_argument(
        '--reconstruction-extent',
        type=ut.check_positive_int, default=20,
        help='Reconstruction extent in pixels (default is 20, range is '
             ']0, +inf[).')

    detection_medial = parser_detection.add_argument_group('Medial axis')
    detection_medial.add_argument(
        '--smoothing', type=ut.check_positive_int, default=20,
        help='Smoothing of the output fibers (default is 20, range is is '
             ']0, +inf[).')
    detection_medial.add_argument(
        '--fibers-minimal-length', type=ut.check_positive_int, default=30,
        help='Minimal length of a fiber in pixels default is 30, range is '
             ']0, +inf[).')
    detection_medial.add_argument(
        '--output', type=ut.check_valid_path, default=None,
        help='Output path for saving detected fibers (default is None).')

    # extraction command
    parser_extraction = subparsers.add_parser(
        'extract',
        help='extract detected fibers from images',
        description='Extract and unfold detected fibers and their associated '
                    'profiles.')
    parser_extraction.set_defaults(func=extraction_command)

    parser_extraction.add_argument(
        'input', type=ut.check_valid_path,
        help='Path to single image file or path to folder with multiple '
             'images.')
    parser_extraction.add_argument(
        'fibers', type=ut.check_valid_directory,
        help='Path to directory containing fiber files.')
    parser_extraction.add_argument(
        '--radius', type=ut.check_positive_int, default=5,
        help='Radius of the fibers to extract from images (default is 5).')
    parser_extraction.add_argument(
        '--group-fibers', action='store_true',
        help='Group the extracted fibers by image.')
    parser_extraction.add_argument(
        '--output', type=ut.check_valid_path, default=None,
        help='Output path for saving profiles and extracted fibers (default '
             'is None).')
    parser_extraction.add_argument(
        '--profiles-only', action='store_true',
        help='Output only the profiles (not figures).')
    parser_extraction.add_argument(
        '--pixel-size', type=float, default=1,
        help='Set the pixel size to the given size in any unit. By default the '
             'pixel size is 1 (no calibration).')

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
        '--output-model', type=str, default=None,
        help='Output path for saving the model (default is None).')

    group_data = parser_analysis.add_argument_group('Quantification')
    group_data.add_argument(
        '--scheme', type=str, nargs='+',
        default=['experiment', 'image', 'fiber'],
        help='Names of the keys used as indexing of the results (default is '
             'experiment, image, fiber; there should be at least one name).')
    group_data.add_argument(
        '--output', type=str, default=None,
        help='Output path for saving data analysis (default is None).')

    # quantification command
    parser_quantification = subparsers.add_parser(
        'quantify',
        help='quantify a detailed analysis',
        description='Export relevant quantification from detailed analysis.')
    parser_quantification.set_defaults(func=quantification_command)
    parser_quantification.add_argument(
        'input', type=ut.check_valid_file,
        help='Input detailed analysis to quantify')
    parser_quantification.add_argument(
        '--output', type=ut.check_valid_output_file, default=None,
        help='Path where output will be written.')
    parser_quantification.add_argument(
        '--scheme', type=str, nargs='+',
        default=['experiment', 'image', 'fiber'],
        help='Names of the keys used as indexing of the results (default is '
             'experiment, image, fiber; there should be at least one name).')

    # simulation command
    parser_simulation = subparsers.add_parser(
        'simulate',
        help='simulate an image containing fibers',
        description='Simulate a single confocal section containing DNA fibers.')
    parser_simulation.set_defaults(func=simulate_command)

    parser_simulation.add_argument(
        '--output', type=ut.check_valid_output_file, default=None,
        help='Output path for saving simulation (default: None).')
    parser_simulation.add_argument(
        '--no-image', action='store_true',
        help='Setting this flag will lead to no image degradation output.')
    parser_simulation.add_argument(
        '--simulated-fibers', type=ut.check_valid_or_empty_path, default='',
        help='Input already simulated fibers.')
    parser_simulation.add_argument(
        '--paths-only', action='store_true',
        help='Setting this flag will lead to no fibers paths and no image '
             'degradation output.')

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
        help='Local perturbations of the fiber path (default is [0.1, 0.3]).')
    fibers_group.add_argument(
        '--bending-elasticity-range', type=float, nargs=2, default=[1, 3],
        help='Elasticity of the global bending of the fiber path (default is '
             '[1, 3]).')
    fibers_group.add_argument(
        '--bending-force-range', type=float, nargs=2, default=[10, 30],
        help='Force of the global bending of the fiber path (default is '
             '[10, 30]).')
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
        '--local-force-range', type=float, nargs=2, default=[0.1, 0.4],
        help='Force of local signal inhomogeneity (default is [0.1, 0.4]).')
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
        '--z-index', type=float, nargs=2, default=[-1, 1],
        help='Z-index of fiber objects (default: [-1, 1]).')
    image_group.add_argument(
        '--snr', type=float, default=7, help='SNR in decibels (default: 7).')

    # comparison command
    parser_comparison = subparsers.add_parser(
        'compare',
        help='compare fibers and analysis results',
        description='Useful to compare expected and actual results of both '
                    'detection and analysis steps.')
    comparison_subparsers = parser_comparison.add_subparsers(
        title='available sub-commands', dest='sub-command')
    comparison_subparsers.required = True

    # comparison-fibers sub-command
    comparison_fibers = comparison_subparsers.add_parser(
        'fibers',
        help='compare fibers',
        description='Compare expected and actual results of detection.')
    comparison_fibers.set_defaults(func=compare_fibers_command)

    comparison_fibers.add_argument(
        'expected', type=ut.check_valid_path,
        help='Expected input to be compared; either a single file (one fiber),'
             'a zip file (multiple fibers) or a directory (multiple fibers).')
    comparison_fibers.add_argument(
        'actual', type=ut.check_valid_path,
        help='Actual input to be compared; either a single file (one fiber),'
             'a zip file (multiple fibers) or a directory (multiple fibers).')
    comparison_fibers.add_argument(
        '--output', type=ut.check_valid_output_file, default=None,
        help='Path of output file in which the comparison results will be '
             'writen.')
    comparison_fibers.add_argument(
        '--scheme', type=str, nargs='+',
        default=['experiment', 'image', 'fiber'],
        help='Names of the keys used as indexing of the results (default is '
             'experiment, image, fiber; there should be at least one name).')

    # comparison-analysis sub-command
    comparison_analyses = comparison_subparsers.add_parser(
        'analyses',
        help='compare analysis',
        description='Compare expected and actual results of the second part of '
                    'the pipeline, the analysis.')
    comparison_analyses.set_defaults(func=comparison_analyses_command)

    comparison_analyses.add_argument(
        'expected', type=ut.check_valid_file,
        help='Expected input to be compared; it must be a valid file.')
    comparison_analyses.add_argument(
        'actual', type=ut.check_valid_file,
        help='Actual input to be compared; it must be a valid file.')
    comparison_analyses.add_argument(
        'match', type=ut.check_valid_file,
        help='Fibers match, generated by the compare fibers command. It must '
             'contain at least expected and actual fiber columns.')
    comparison_analyses.add_argument(
        '--output', type=ut.check_valid_output_file, default=None,
        help='Path of output file in which the comparison results will be '
             'writen.')
    comparison_analyses.add_argument(
        '--scheme', type=str, nargs='+',
        default=['experiment', 'image', 'fiber'],
        help='Names of the keys used as indexing of the results (default is '
             'experiment, image, fiber; there should be at least one name).')

    # dataset creation sub-command
    parser_dataset = subparsers.add_parser(
        'create-dataset',
        help='create a new dataset zip file from paths',
        description='This command is used to create dataset with for instance '
                    'manual annotations for benchmarking purposes.')
    parser_dataset.set_defaults(func=create_dataset)
    parser_dataset.add_argument(
        'summary', type=ut.check_valid_file,
        help='Path to the summary csv file.')
    parser_dataset.add_argument(
        'images', type=ut.check_valid_directory,
        help='Path to the images.')
    parser_dataset.add_argument(
        'fibers', type=ut.check_valid_directory,
        help='Path to the fibers.')
    parser_dataset.add_argument(
        'profiles', type=ut.check_valid_directory,
        help='Path to the profiles')
    parser_dataset.add_argument(
        '--output', type=ut.check_valid_output_file,
        default='./dataset.zip',
        help='Path to the output file (the zip file containing the dataset).')

    # parsing and dispatch
    args = parser.parse_args()

    if args.batch:
        from matplotlib import pyplot as plt
        plt.switch_backend('Agg')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        args.func(args)

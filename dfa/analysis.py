"""
Analysis module of the DNA fiber analysis package.

Use this module to analyse fiber profiles (extracted with the detection
module).

Note that for patterns with only one segment (no splits), the results might
be biased since then we have to rely on the intensity ratio, instead of the
intensity ratio derivative, to choose the channels pattern.
"""
import numpy as np
import pandas as pd
import copy

from dfa import modeling, _tree


def _select_possible_patterns(x, y, model=modeling.standard,
                              min_length=4,
                              error_func=lambda v1, v2: np.power(v1-v2,
                                                                 2).sum()):
    """
    Select the matching models.

    The models are chosen according to the modeling given. The output patterns
    are ordered in increasing error (given by the error function). Any pattern
    that does not have a binary alternative scheme will be dropped.

    :param x: Independent variables.
    :type x: numpy.ndarray

    :param y: Dependent variables.
    :type y: numpy.ndarray

    :param model: Model defining the patterns to detect and filter (default is
    the standard model defined in dfa.modeling.standard).
    :type model: dfa.modeling.Model

    :param min_length: Minimal length of a segment. Default is 4 pixels, which
    corresponds to the thickness of a fiber when pixel size is equal to
    0.1419761 microns.
    :type min_length: strictly positive integer

    :param error_func: function used to quantify the quality of the patterns
    (default is the residuals-sum of squared errors).
    :type: function

    :return: For each possible pattern, the error, the splits and the channels
    patterns.
    :rtype: list of tuple
    """
    def _alternate_constraint_func(node):
        if node is node.parent.left:
            return (node.values[4] - node.values[3]) * \
                   (node.parent.values[4] - node.values[4]) < 0
        else:
            return (node.values[3] - node.parent.values[3]) * \
                   (node.values[4] - node.values[3]) < 0

    def _leave_one_out(x, y, max_partitions):
        error = 0

        for i in range(x.size):
            select = np.array(range(x.size)) != i
            reg = _tree.RegressionTree()
            reg = reg.fit(x[select], y[select])
            error += error_func(
                y[i], reg.predict(x[i],
                                  constraint_func=_alternate_constraint_func,
                                  max_partitions=max_partitions))

        return error

    selected_patterns = []

    reg = _tree.RegressionTree(min_samples=min_length)
    reg = reg.fit(x, y)

    # Models can be symmetric
    channels_patterns = model.channels_patterns()
    channels_patterns += [channels_pattern[::-1]
                          for channels_pattern in channels_patterns]

    for number_of_segments in model.numbers_of_segments():
        prediction_y = reg.predict(x, max_partitions=number_of_segments,
                                   constraint_func=_alternate_constraint_func)
        prediction_diff = np.diff(prediction_y)
        splits = np.where(prediction_diff != 0)[0].tolist()

        # Check first the alternative scheme
        if all(prediction_diff[splits[i]] * prediction_diff[splits[i+1]] < 0
               for i in range(len(splits)-1)):
            # Create segments pattern from splits
            channels_pattern = list(np.less(prediction_diff[splits], 0)
                                    .astype('int').tolist())

            if len(channels_pattern) > 0:
                channels_pattern.append(1 - channels_pattern[-1])
            else:  # When no split, the resulting pattern rely on the intensity
                channels_pattern.append(int(prediction_y[0] > 0))

            # Check if pattern is in model (and symmetric)
            if channels_pattern in channels_patterns:
                # Use cross-validation (leave-one-out) to compute error
                selected_patterns.append((_leave_one_out(x, y,
                                                         number_of_segments),
                                          splits, channels_pattern))

    return selected_patterns


def _choose_pattern(selected_patterns, x, y, discrepancy, contrast):
    """
    Choose the best pattern with given criterion.

    :param selected_patterns: Possible patterns.
    :type selected_patterns: list of tuples

    :param discrepancy: Factor of discrepancy regularization between amplitudes
    of the same marker.
    :type discrepancy: positive float

    :param contrast: Factor of contrast regularization between amplitudes of
    opposite markers.
    :type contrast: positive float

    :return: Chosen pattern
    :rtype: tuple
    """
    # NOTE maybe add also a model complexity regularization (branches)?
    def _mean(l):
        return sum(l) / len(l) if len(l) > 0 else 0

    def _var(l, m):
        return sum([(e - m) ** 2 for e in l])

    def _regularize_patterns(patterns, discrepancy, contrast):
        regularized_patterns = []

        for error, splits, pattern in patterns:
            if len(pattern) > 1:
                bounds = copy.copy(splits)
                bounds.insert(0, 0)
                bounds.append(x.size - 1)

                constants = ([], [])
                for i in range(len(pattern)):
                    constants[pattern[i]].append(
                        y[bounds[i]:bounds[i + 1]].mean())

                means = (_mean(constants[0]), _mean(constants[1]))

                contrast_regularization = 1 / (means[0] - means[1]) ** 2
                discrepancy_regularization = \
                    (_var(constants[0], means[0]) +
                     _var(constants[1], means[1])) \
                    / (len(pattern) - 1)
            else:
                contrast_regularization = 0
                discrepancy_regularization = 0

            error += \
                discrepancy * discrepancy_regularization + \
                contrast * contrast_regularization

            regularized_patterns.append((error, splits, pattern))

        return regularized_patterns

    regularized_patterns = _regularize_patterns(
        selected_patterns, discrepancy, contrast)

    regularized_patterns.sort(key=lambda e: e[0])
    return regularized_patterns[0]


def analyze(profile, model=modeling.standard, channels_names=('CIdU', 'IdU'),
            min_length=4, discrepancy=0, contrast=0):
    """
    Detect the segments in profile and analyze it.

    By default, it takes the model with the minimal error.

    :param profile: Input profile (containing the X values and the Y values of
    the two channels as column vectors of a matrix).
    :type profile: numpy.ndarray

    :param model: Model defining the patterns to detect and filter (default is
    the standard model defined in dfa.modeling.standard).
    :type model: dfa.modeling.Model

    :param channels_names: Names of the channels in the same order as they
    appear in the profile.
    :type channels_names: tuple of str of size 2

    :param min_length: Minimal length of a segment. Default is 4 pixels, which
    corresponds to the thickness of a fiber when pixel size is equal to
    0.1419761 microns.
    :type min_length: strictly positive integer

    :param discrepancy: Factor of discrepancy regularization between amplitudes
    of the same marker.
    :type discrepancy: positive float

    :param contrast: Factor of contrast regularization between amplitudes of
    opposite markers.
    :type contrast: positive float

    :return: A reference to a pattern defined in model and the lengths.
    :rtype: list of dict and list or None (if no pattern is found)

    :raises ValueError: In case inputs are not valid.
    """
    if type(profile) != np.ndarray:
        raise TypeError('Input profile must be of type numpy.ndarray!\n'
                        'It is of type {}...'.format(type(profile)))

    if profile.shape[0] <= 1 or profile.shape[1] != 3:
        raise ValueError('Input profile must have a shape equal to Nx3 '
                         '(N>=1 rows and 3 columns)!\n'
                         'It has shape equal to {}...'.format(profile.shape))

    if type(model) != modeling.Model:
        raise TypeError('Input model must by of type dfa.modeling.Model!\n'
                        'It is of type {}...'.format(type(model)))

    if type(channels_names) != tuple and type(channels_names) != list:
        raise TypeError('Input channels names must be of type tuple or list!\n'
                        'It is of type {}...'.format(type(channels_names)))

    if len(channels_names) != 2:
        raise ValueError('Input channels names must have size equal to 2\n'
                         'The number of channels is limited to 2.')

    if type(min_length) != int:
        raise TypeError('Minimal length parameter must be of type int!\n'
                        'It is of type {}...'.format(type(min_length)))

    if min_length <= 0:
        raise ValueError('Minimal length parameter must be strictly '
                         'positive!\nIt is equal to {}...'.format(min_length))

    channels_indices = [1 + model.channels_names.index(cn)
                        for cn in channels_names]

    x = profile[:, 0]
    y1, y2 = profile[:, channels_indices[1]], profile[:, channels_indices[0]]
    y = np.log(y1) - np.log(y2)
    possible_patterns = _select_possible_patterns(
        x, y, model=model, min_length=min_length)

    _, splits, channels_pattern = _choose_pattern(
        possible_patterns, x, y, discrepancy=discrepancy, contrast=contrast)
    splits.insert(0, 0)
    splits.append(x.size-1)
    lengths = np.diff(x[splits])

    pattern = model.search(channels_pattern)

    # Handle the symmetric case for match the pattern in the same order
    if pattern is None:
        pattern = model.search(channels_pattern[::-1])
        lengths = lengths[::-1]

    return pattern, lengths


def analyzes(profiles, model=modeling.standard, update_model=True, keys=None,
             keys_names=None, channels_names=('CIdU', 'IdU'),
             discrepancy=0, contrast=0):
    """
    Detect the segments in each profile and analyze it.

    Internally, it loops over the profiles and use the analyze function.

    :param profiles: Input profiles to analyze.
    :type profiles: list

    :param model: Model defining the patterns to detect and filter (default is
    the standard model defined in dfa.modeling.standard).
    :type model: dfa.modeling.Model

    :param update_model: Flag to update the model or not (default is True). If
    model is updated, it is then possible to extract frequencies of patterns and
    mean and std lengths.
    :type update_model: bool

    :param keys: A list of tuples to use as key index of rows for profiles'
    results (default is None). Each key must have the same size as the keys
    names.
    :type keys: list of tuples

    :param keys_names: A list of strings to use as columns headers for indexing
    columns (default is None). The list must have the same size as each key.
    :type keys_names: list of str

    :param channels_names: Names of the channels in the same order as they
    appear in the profile.
    :type channels_names: tuple of str of size 2

    :param discrepancy: Factor of discrepancy regularization between amplitudes
    of the same marker.
    :type discrepancy: positive float

    :param contrast: Factor of contrast regularization between amplitudes of
    opposite markers.
    :type contrast: positive float

    :return: A data structure containing the detailed measurements.

    :raises ValueError: In case inputs are not valid.
    """
    if type(profiles) != list:
        raise TypeError('Input profiles must be a list of profiles!\n'
                        'It is of type {}...'.format(type(profiles)))

    if type(model) != modeling.Model:
        raise TypeError('Input model must by of type dfa.modeling.Model!\n'
                        'It is of type {}...'.format(type(model)))

    if type(update_model) != bool:
        raise TypeError('Update model flag must be of type bool!\n'
                        'It is of type {}...'.format(type(update_model)))

    if keys is not None:
        if type(keys) != list:
            raise TypeError('Index must be of type list!\n'
                            'It is of type {}...'.format(type(keys)))

        if len(keys) != len(profiles):
            raise TypeError('Index and profiles must have the same size!\n'
                            'Index has {} and profiles '
                            'has {}...'.format(len(keys), len(profiles)))

        if any(type(key) != tuple for key in keys):
            raise TypeError('Key index must be of type tuple!\n'
                            'At least one key is not of type tuple...')

        if any(len(key) != len(keys_names) for key in keys):
            raise ValueError('Keys and keys names must have the same size!\n'
                             'At least one key does not have {} elements...'
                             .format(len(keys_names)))

        index = pd.MultiIndex(levels=[[] for _ in range(len(keys_names))],
                              labels=[[] for _ in range(len(keys_names))],
                              names=keys_names)
    else:
        keys = range(len(profiles))
        index = pd.MultiIndex(levels=[[]], labels=[[]],
                              names=['profile'])

    labels = ['pattern', 'channel', 'length']
    detailed_analysis = pd.DataFrame([], columns=labels, index=index)

    for key, profile in zip(keys, profiles):
        pattern, lengths = analyze(profile, model=model,
                                   channels_names=channels_names,
                                   discrepancy=discrepancy, contrast=contrast)
        model.append_sample(pattern, lengths)

        for length, channel in zip(lengths, pattern['channels']):
            s = pd.Series({labels[0]: pattern['name'],
                           labels[1]: model.channels_names[channel],
                           labels[2]: length},
                          name=key)
            detailed_analysis = detailed_analysis.append(s)

    if update_model:
        model.update_model()

    return detailed_analysis


def fork_speed(data, channel='CIdU', pattern_name='ongoing fork',
               kb_per_microns=2.5):
    """
    Calculate fork speeds from a detailed analysis.

    :param data: Detailed analysis of DNA fibers.
    :type data: pandas.DataFrame

    :param channel: Name of the channel to consider (default is 'CIdU').
    :type channel: str

    :param pattern_name: Name of the pattern to consider (default is 'ongoing
    fork').
    :type pattern_name: str

    :param kb_per_microns: Number of kb per microns along the DNA fibers.
    :type kb_per_microns: strictly positive float

    :return: The calculated fork speeds in kb.
    :rtype: list of float

    :raises ValueError: In case inputs are not valid.
    """
    if type(data) != pd.DataFrame:
        raise TypeError('The data type must be pandas.DataFrame!\n'
                        'It is of type {}...'.format(type(data)))

    if 'pattern' not in data.columns:
        raise ValueError('The data frame must contain a column '
                         'named "pattern"!')

    if 'channel' not in data.columns:
        raise ValueError('The data frame must contain a column '
                         'named "channel"!')

    if 'length' not in data.columns:
        raise ValueError('The data frame must contain a column '
                         'named "length"!')

    if type(channel) != str:
        raise TypeError('The type of channel must be str!\n'
                        'It is of type {}...'.format(type(channel)))

    if type(pattern_name) != str:
        raise TypeError('The type of pattern_name must be str!\n'
                        'It is of type {}...'.format(type(pattern_name)))

    if type(kb_per_microns) != float:
        raise TypeError('The type of kb_per_microns must be float!\n'
                        'It is of type {}...'.format(type(kb_per_microns)))

    if kb_per_microns <= 0:
        raise ValueError('The kb_per_microns variable must be strictly'
                         ' greater than {}!'.format(kb_per_microns))

    # FIXME kb per microns ?

    subset = data[data['pattern'] == pattern_name]

    if not subset.empty:
        return data[data['channel'] == channel].ix[
            subset.index.unique(), 'length'].tolist()
    else:
        return []


def fork_rate(data, channel='CIdU', pattern_name='1st label origin'):
    """
    Calculate fork rates from a detailed analysis.

    :param data: Detailed analysis of DNA fibers.
    :type data: pandas.DataFrame

    :param channel: Name of the channel to consider (default is 'CIdU').
    :type channel: str

    :param pattern_name: Name of the pattern to consider (default is 'ongoing
    fork').
    :type pattern_name: str

    :return: The calculated fork rates.
    :rtype: list of float

    :raises ValueError: In case inputs are not valid.
    """
    if type(data) != pd.DataFrame:
        raise TypeError('The data type must be pandas.DataFrame!\n'
                        'It is of type {}...'.format(type(data)))

    if 'pattern' not in data.columns:
        raise ValueError('The data frame must contain a column '
                         'named "pattern"!')

    if 'channel' not in data.columns:
        raise ValueError('The data frame must contain a column '
                         'named "channel"!')

    if 'length' not in data.columns:
        raise ValueError('The data frame must contain a column '
                         'named "length"!')

    if type(channel) != str:
        raise TypeError('The type of channel must be str!\n'
                        'It is of type {}...'.format(type(channel)))

    if type(pattern_name) != str:
        raise TypeError('The type of pattern_name must be str!\n'
                        'It is of type {}...'.format(type(pattern_name)))

    fork_rates = []

    subset = data[data['pattern'] == pattern_name]

    if not subset.empty:
        for index in subset.index.unique():
            values = data[data['channel'] == channel].ix[index, 'length']
            fork_rates.append(values.max() / values.min())

    return fork_rates


if __name__ == '__main__':
    import os
    import copy
    import argparse

    # Parse input arguments
    parser = argparse.ArgumentParser()

    group_profile = parser.add_argument_group('Profiles')
    group_profile.add_argument('input', type=str,
                               help='Input path to profile(s) (folder or file).'
                                    ' Profiles are assumed to have at least 3 '
                                    'columns, the first one being the values '
                                    'of the x-axis.')
    group_profile.add_argument('--channels_names', type=str, nargs='+',
                               default=['CIdU', 'IdU'],
                               help='Names of the channels as they appear in '
                                    'order in the profiles (default is CIdU '
                                    'and IdU).')
    group_profile.add_argument('--input_columns', type=int, nargs='+',
                               default=[1, 2],
                               help='Columns index of the profiles to use')
    group_profile.add_argument('--recursive', action='store_true',
                               help='Search in specified path recursively '
                                    '(default is False; works only for '
                                    'directory input).')

    group_model = parser.add_argument_group('Model')
    group_model.add_argument('--model', type=str, default=None,
                             help='Path to the model to use (default will use '
                                  'the standard model defined in the '
                                  'dfa.modeling module).')
    group_model.add_argument('--discrepancy', type=float, default=0,
                             help='Discrepancy regularization on intensity of '
                                  'branches of the same channel (default is '
                                  '0, i.e. deactivated).')
    group_model.add_argument('--contrast', type=float, default=0,
                             help='Contrast regularization between intensities '
                                  'of branches of opposite channels (default '
                                  'is 0, i.e. deactivated).')
    group_model.add_argument('--output_model', type=str, default=None,
                             help='Output path for saving the model (default '
                                  'is None).')

    group_data = parser.add_argument_group('Quantification')
    group_data.add_argument('--scheme', type=str, nargs='+',
                            default=['experiment', 'image', 'fiber'],
                            help='Names of the keys used as indexing of the '
                                 'results (default is experiment, image, '
                                 'fiber; there should be at least one name).')
    group_data.add_argument('--keys_in_file', type=str, default=None,
                            help='If set, the keys are searched in the '
                                 'filenames (separator must be provided); '
                                 'otherwise the keys are searched in the last '
                                 'path elements (folders and filenames '
                                 'separated by /).')
    group_data.add_argument('--output', type=str, default=None,
                            help='Output path for saving data analysis '
                                 '(default is None).')

    args = parser.parse_args()

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
                           usecols=[0]+args.input_columns)
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
        model = copy.deepcopy(modeling.standard)
    else:
        if not os.path.isfile(args.model):
            raise ValueError('The input model argument must be a valid path'
                             ' to a text file!')

        model = modeling.Model.load(args.model)

    model.initialize_model()
    detailed_analysis = analyzes(
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

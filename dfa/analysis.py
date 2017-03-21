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

import _tree
import modeling


def _select_possible_patterns(x, y, model=modeling.standard,
                              error_func=lambda v1, v2: np.power(v1-v2,
                                                                 2).sum(),
                              min_error_improvement=0.05):
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

    :param error_func: function used to quantify the quality of the patterns
    (default is the residuals-sum of squared errors).
    :type: function

    :param min_error_improvement: Minimum error improvement necessary for a
    given model to be kept. Default is 0.05 (%5 of the maximum error).
    :type min_error_improvement: float

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

    selected_patterns = []

    reg = _tree.RegressionTree()
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
            channels_pattern = list(np.less(prediction_diff[splits], 0)
                                    .astype('int').tolist())

            if len(channels_pattern) > 0:
                channels_pattern.append(1 - channels_pattern[-1])
            else:  # When no split, the resulting pattern rely on the intensity
                channels_pattern.append(int(prediction_y[0] > 0))

            # Check if pattern is in model (and symmetric)
            if channels_pattern in channels_patterns:
                selected_patterns.append((error_func(y, prediction_y),
                                          splits, channels_pattern))

    selected_patterns.sort(key=lambda e: e[0])

    errors = np.array(list(zip(*selected_patterns))[0])
    indices_to_remove = np.where(np.diff(errors / errors.max()) <
                                 min_error_improvement)[0].tolist()

    filtered_patterns = selected_patterns.copy()

    for index in indices_to_remove:
        filtered_patterns.remove(selected_patterns[index])

    return filtered_patterns


def analyze(profile, model=modeling.standard, channels_names=('CIdU', 'IdU'),
            min_error_improvement=0.05):
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

    :param min_error_improvement: Minimum error improvement necessary for a
    given model to be kept, as a percentage (in range [0,1]). Default is 0.05
    (%5 of the maximum error).
    :type min_error_improvement: float

    :return: A reference to a pattern defined in model and the lengths.
    :rtype: list of dict and list or None (if no pattern is found)

    :raises ValueError: In case inputs are not valid.
    """
    if type(profile) != np.ndarray:
        raise ValueError('Input profile must be of type numpy.ndarray!\n'
                         'It is of type {}...'.format(type(profile)))

    if profile.shape[0] <= 1 or profile.shape[1] != 3:
        raise ValueError('Input profile must have a shape equal to Nx3 '
                         '(N>=1 rows and 3 columns)!\n'
                         'It has shape equal to {}...'.format(profile.shape))

    if type(model) != modeling.Model:
        raise ValueError('Input model must by of type dfa.modeling.Model!\n'
                         'It is of type {}...'.format(type(model)))

    if type(channels_names) != tuple and type(channels_names) != list:
        raise ValueError('Input channels names must be of type tuple or list!\n'
                         'It is of type {}...'.format(type(channels_names)))

    if len(channels_names) != 2:
        raise ValueError('Input channels names must have size equal to 2\n'
                         'The number of channels is limited to 2.')

    if type(min_error_improvement) != float:
        raise ValueError('Minimum error improvement parameter must be '
                         'of type float!\nIt is of type {}...'
                         .format(type(min_error_improvement)))

    if min_error_improvement < 0 or min_error_improvement > 1:
        raise ValueError('Minimum error improvement parameter must be in '
                         'range [0,1]!\nIt is {}...'
                         .format(min_error_improvement))

    channels_indices = [1 + model.channels_names.index(cn)
                        for cn in channels_names]

    x = profile[:, 0]
    y1, y2 = profile[:, channels_indices[1]], profile[:, channels_indices[0]]
    y = np.log(y1) - np.log(y2)
    possible_patterns = _select_possible_patterns(
        x, y, model=model, min_error_improvement=min_error_improvement)

    _, splits, channels_pattern = possible_patterns[0]
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
             min_error_improvement=0.05):
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

    :param min_error_improvement: Minimum error improvement necessary for a
    given model to be kept. Default is 0.05 (%5 of the maximum error).
    :type min_error_improvement: float

    :return: A data structure containing the detailed measurements.

    :raises ValueError: In case inputs are not valid.
    """
    if type(profiles) != list:
        raise ValueError('Input profiles must be a list of profiles!\n'
                         'It is of type {}...'.format(type(profiles)))

    if type(model) != modeling.Model:
        raise ValueError('Input model must by of type dfa.modeling.Model!\n'
                         'It is of type {}...'.format(type(model)))

    if type(update_model) != bool:
        raise ValueError('Update model flag must be of type bool!\n'
                         'It is of type {}...'.format(type(update_model)))

    if keys is not None:
        if type(keys) != list:
            raise ValueError('Index must be of type list!\n'
                             'It is of type {}...'.format(type(keys)))

        if len(keys) != len(profiles):
            raise ValueError('Index and profiles must have the same size!\n'
                             'Index has {} and profiles '
                             'has {}...'.format(len(keys), len(profiles)))

        if any(type(key) != tuple for key in keys):
            raise ValueError('Key index must be of type tuple!\n'
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
                                   min_error_improvement=min_error_improvement)
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
        raise ValueError('The data type must be pandas.DataFrame!\n'
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
        raise ValueError('The type of channel must be str!\n'
                         'It is of type {}...'.format(type(channel)))

    if type(pattern_name) != str:
        raise ValueError('The type of pattern_name must be str!\n'
                         'It is of type {}...'.format(type(pattern_name)))

    if type(kb_per_microns) != float:
        raise ValueError('The type of kb_per_microns must be float!\n'
                         'It is of type {}...'.format(type(kb_per_microns)))

    if kb_per_microns <= 0:
        raise ValueError('The kb_per_microns variable must be strictly'
                         ' greater than {}!'.format(kb_per_microns))

    return data[data['channel'] == channel].ix[
        data[data['pattern'] == pattern_name].index.unique(),
        'length'].tolist()


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
        raise ValueError('The data type must be pandas.DataFrame!\n'
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
        raise ValueError('The type of channel must be str!\n'
                         'It is of type {}...'.format(type(channel)))

    if type(pattern_name) != str:
        raise ValueError('The type of pattern_name must be str!\n'
                         'It is of type {}...'.format(type(pattern_name)))

    fork_rates = []

    for index in data[data['pattern'] == pattern_name].index.unique():
        values = data[data['channel'] == channel].ix[index, 'length']
        fork_rates.append(values.max() / values.min())

    return fork_rates


if __name__ == '__main__':
    import os
    import copy
    import argparse

    def check_min_error_improvement(s):
        """ Range checking for minimum error improvement value. """
        value = float(s)

        if value < 0 or value > 1:
            raise argparse.ArgumentTypeError('The value of the minimum error'
                                             ' improvement argument must be '
                                             'within range [0, 1]!')

        return value

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
    group_model.add_argument('--min_error_improvement', default=0.05,
                             type=check_min_error_improvement,
                             help='Minimum error improvement used to filter '
                                  'out unnecessary complex models (default is '
                                  '0.05, acceptable range is [0,1]).')
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
        min_error_improvement=args.min_error_improvement,
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

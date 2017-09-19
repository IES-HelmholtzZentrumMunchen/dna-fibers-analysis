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
import sys

from dfa import modeling, _tree


def _select_possible_patterns(x, y, model=modeling.standard,
                              min_length=4,
                              error_func=lambda v1, v2: np.power(v1-v2,
                                                                 2).sum()):
    """Select the matching models.

    The models are chosen according to the modeling given. The output patterns
    are ordered in increasing error (given by the error function). Any pattern
    that does not have a binary alternative scheme will be dropped.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variables.

    y : numpy.ndarray
        Dependent variables.

    model : dfa.modeling.Model
        Model defining the patterns to detect and filter (default is the
        standard model defined in dfa.modeling.standard).

    min_length : strictly positive integer
        Minimal length of a segment. Default is 4 pixels, which corresponds to
        the thickness of a fiber when pixel size is equal to 0.1419761 microns.

    error_func : callable function
        function used to quantify the quality of the patterns (default is the
        residuals-sum of squared errors).

    Returns
    -------
    list of tuple
        For each possible pattern, the error, the splits and the channels
        patterns.
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
    """Choose the best pattern with given criterion.

    Parameters
    ----------
    selected_patterns : list of tuples
        Possible patterns.

    discrepancy : positive float
        Factor of discrepancy regularization between amplitudes of the same
        marker.

    contrast : positive float
        Factor of contrast regularization between amplitudes of opposite
        markers.

    Returns
    -------
    (float, List[int], List[int])
        Chosen pattern
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
    """Detect the segments in profile and analyze it.

    By default, it takes the model with the minimal error.

    Parameters
    ----------
    profile : numpy.ndarray
        Input profile (containing the X values and the Y values of the two
        channels as column vectors of a matrix).

    model : dfa.modeling.Model
        Model defining the patterns to detect and filter (default is the
        standard model defined in dfa.modeling.standard).

    channels_names : tuple of str of size 2
        Names of the channels in the same order as they appear in the profile.

    min_length : strictly positive integer
        Minimal length of a segment. Default is 4 pixels, which corresponds to
        the thickness of a fiber when pixel size is equal to 0.1419761 microns.

    discrepancy : positive float
        Factor of discrepancy regularization between amplitudes of the
        same marker.

    contrast : positive float
        Factor of contrast regularization between amplitudes of opposite
        markers.

    Returns
    -------
    list of dict and list or None (if no pattern is found)
        A reference to a pattern defined in model and the lengths.

    Raises
    ------
    ValueError
        In case inputs are not valid.

    See Also
    --------
    analyzes : Analyze profiles from multiple fibers.
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

    # the + 1 is here to avoid nan due to zero values
    x = profile[:, 0]
    y1 = profile[:, channels_indices[1]] + 1
    y2 = profile[:, channels_indices[0]] + 1
    y = np.log(y1) - np.log(y2)
    possible_patterns = _select_possible_patterns(
        x, y, model=model, min_length=min_length)

    if len(possible_patterns) > 0:
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
    else:
        return None, None


def analyzes(profiles, model=modeling.standard, update_model=True, keys=None,
             keys_names=None, channels_names=('CIdU', 'IdU'),
             discrepancy=0, contrast=0):
    """Detect the segments in each profile and analyze it.

    Internally, it loops over the profiles and use the analyze function.

    Parameters
    ----------
    profiles : List[numpy.ndarray]
        Input profiles to analyze.

    model : dfa.modeling.Model
        Model defining the patterns to detect and filter (default is the
        standard model defined in dfa.modeling.standard).

    update_model : bool
        Flag to update the model or not (default is True). If model is updated,
        it is then possible to extract frequencies of patterns and mean and
        std lengths.

    keys : None | List[(T, U, ...)]
        A list of tuples to use as key index of rows for profiles' results
        (default is None). Each key must have the same size as the keys names.

    keys_names : List[str]
        A list of strings to use as columns headers for indexing columns
        (default is None). The list must have the same size as each key.

    channels_names : tuple of str of size 2
        Names of the channels in the same order as they appear in the profile.

    discrepancy : positive float
        Factor of discrepancy regularization between amplitudes of the
        same marker.

    contrast : positive float
        Factor of contrast regularization between amplitudes of opposite
        markers.

    Returns
    -------
    pandas.DataFrame
        A data structure containing the detailed measurements.

    Raises
    ------
    ValueError
        In case inputs are not valid.

    See Also
    --------
    analyze : Analyze profiles from a single fiber.
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
        try:
            pattern, lengths = analyze(profile, model=model,
                                       channels_names=channels_names,
                                       discrepancy=discrepancy,
                                       contrast=contrast)

            if pattern is not None and lengths is not None:
                model.append_sample(pattern, lengths)

                for length, channel in zip(lengths, pattern['channels']):
                    s = pd.Series({labels[0]: pattern['name'],
                                   labels[1]: model.channels_names[channel],
                                   labels[2]: length},
                                  name=key)
                    detailed_analysis = detailed_analysis.append(s)
        except AttributeError:
            print('----> Error during analysis! Omitting {}!'.format(key),
                  file=sys.stderr)

    if update_model:
        model.update_model()

    return detailed_analysis


def fork_speed(data, channel='CIdU', pattern_name='ongoing fork',
               kb_per_microns=2.5):
    """
    Calculate fork speeds from a detailed analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        Detailed analysis of DNA fibers.

    channel : str
        Name of the channel to consider (default is 'CIdU').

    pattern_name : str
        Name of the pattern to consider (default is 'ongoing fork').

    kb_per_microns : 0 < float
        Number of kb per microns along the DNA fibers.

    Returns
    -------
    pandas.Series
        The calculated fork speeds in kb for each selected fiber.

    Raises
    ------
    ValueError
        In case inputs are not valid.

    See Also
    --------
    fork_rate : Compute the fork rate of the analyzed fibers.
    get_patterns : Get the patterns per fiber.
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
        s = data[data['channel'] == channel].ix[
            subset.index.unique()]['length']
    else:
        s = pd.Series()

    s.name = 'Fork speed'
    return s


def fork_rate(data, channel='CIdU', pattern_name='1st label origin'):
    """
    Calculate fork rates from a detailed analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        Detailed analysis of DNA fibers.

    channel : str
        Name of the channel to consider (default is 'CIdU').

    pattern_name : str
        Name of the pattern to consider (default is 'ongoing fork').

    Returns
    -------
    pandas.Series
        The calculated fork rates.

    Raises
    ------
    ValueError
        In case inputs are not valid.

    See Also
    --------
    fork_speed : Compute the fork speed of the analyzed fibers.
    get_patterns : Get the patterns per fiber.
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
            values = data[data['channel'] == channel].ix[index]['length']
            fork_rates.append(values.max() / values.min())

    fork_rates = pd.Series(data=fork_rates, index=subset.index.unique(),
                           name='Fork rate')
    fork_rates.index.names = data.index.names

    return fork_rates


def get_patterns(data):
    """
    Output patterns from detailed analysis (useful for pattern frequency
    analysis).

    Parameters
    ----------
    data : pandas.DataFrame
        Detailed analysis of DNA fibers.

    Returns
    -------
    pandas.Series
        The patterns for each fiber.

    See Also
    --------
    fork_speed : Compute the fork speed of the analyzed fibers.
    fork_rate : Compute the fork rate of the analyzed fibers.
    """
    patterns = data.reset_index()[data.index.names + ['pattern']]\
        .drop_duplicates().set_index(data.index.names)['pattern']
    patterns.name = 'Patterns'

    return patterns

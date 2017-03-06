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
from sklearn.tree import DecisionTreeRegressor
from debtcollector import removals

from dfa import _tree
from dfa import modeling


@removals.remove(removal_version='?')
def _piecewise_constant_regression(x, y, num_pieces):
    """
    Piecewise constant regression is implemented as a regression tree.

    This regression uses the constrained step model via regression trees.

    :param x: Independent variable.
    :type x: Column vector (numpy.ndarray with shape (N,1)).

    :param y: Dependent variable.
    :type y: Column vector (numpy.ndarray with shape (N,1)).

    :param num_pieces: Expected number of pieces forming the model.
    :type num_pieces: Integer greater than or equal to 2.

    :return: The predicted values, the change points and sum of squared errors.
    :rtype: Tuple containing a column vector (numpy.ndarray with shape (N,1)),
    a list of num_pieces-1 integers and a positive floating point number.
    """
    # begin pre-conditions
    assert type(x) == np.ndarray
    assert x.shape[0] > 1 and x.shape[1] == 1

    assert type(y) == np.ndarray
    assert y.shape == x.shape

    assert type(num_pieces) == int
    assert num_pieces > 1
    # end pre-conditions

    tree_regression = DecisionTreeRegressor(max_depth=5,
                                            max_leaf_nodes=num_pieces)
    tree_regression.fit(x, y)

    predict_y = tree_regression.predict(x).reshape(x.size, 1)
    sse = float(np.power(y - predict_y, 2).sum())

    change_points = np.where(np.diff(predict_y, axis=0) != 0)[0].tolist()

    # begin post-conditions
    assert type(predict_y) == np.ndarray
    assert predict_y.shape == x.shape

    assert type(change_points) == list
    assert len(change_points) == num_pieces-1
    assert all(type(change_point) == int for change_point in change_points)

    assert type(sse) == float
    assert sse > 0.
    # end post-conditions

    return predict_y, change_points, sse


@removals.remove(removal_version='?')
def _constant_regression(y):
    """
    Constant regression is a linear constrained regression.

    This is a particular case of the constrained step model.

    :param y: Dependent values.
    :type y: Column vector (numpy.array with shape (N,1)).

    :return: The predicted values, the change points and sum of squared errors.
    :rtype: Tuple containing a column vector (numpy.array with shape (N,1)),
    a list of num_pieces-1 integers and a positive floating point number.
    """
    # begin pre-conditions
    assert type(y) == np.ndarray
    assert y.shape[0] > 1 and y.shape[1] == 1
    # end pre-conditions

    predict_y = np.ones(y.shape) * y.mean()
    sse = float(np.power(y-predict_y, 2).sum())

    # begin post-conditions
    assert type(predict_y) == np.ndarray
    assert predict_y.shape == y.shape

    assert type(sse) == float
    assert sse > 0.
    # end post-conditions

    return predict_y, [], sse


@removals.remove(removal_version='?')
def _choose_piecewise_model(x, y, models=(1, 2, 3)):
    """
    Automatic model selection for piecewise constant model fitting.

    The models with number of pieces in models are fitted to the input data.
    Candidate models are first checked for alternative scheme constraint
    (alternate positive/negative values). Finally, the best model is chosen
    among the ones respecting constraint by their sum of squared error.

    :param x: Independent variable.
    :type x: Column vector (numpy.ndarray with shape (N,1)).

    :param y: Dependent variable.
    :type y: Column vector (numpy.ndarray with shape (N,1)).

    :param models: Models to test defined by their number of pieces.
    :type models: List or tuple of integers.

    :return: The predicted values, the change points and sum of squared errors
    of the best model.
    :rtype: Tuple containing a column vector (numpy.array with shape (N,1)),
    a list of num_pieces-1 integers and a positive floating point number.
    """
    def _check_alternative_constraint(values, indices):
        return all(values[index-1] * values[index+1] < 0 for index in indices)

    # begin pre-conditions
    assert type(x) == np.ndarray
    assert x.shape[0] > 1 and x.shape[1] == 1

    assert type(y) == np.ndarray
    assert y.shape == x.shape

    assert type(models) == list or type(models) == tuple
    assert all(type(model) == int for model in models)
    # end pre-conditions

    results = []

    for model in models:
        if model == 1:
            predict_y, change_points, sse = _constant_regression(y)
        else:
            predict_y, change_points, sse = _piecewise_constant_regression(
                x, y, model)

        if _check_alternative_constraint(predict_y, change_points):
            results.append((predict_y, change_points, sse))

    results.sort(key=lambda result: result[2])
    predict_y, change_points, sse = results[0]

    # begin post-conditions
    assert type(predict_y) == np.ndarray
    assert predict_y.shape == x.shape

    assert type(change_points) == list
    assert all(type(change_point) == int for change_point in change_points)

    assert type(sse) == float
    assert sse > 0.
    # end post-conditions

    return predict_y, change_points, sse


def _select_possible_patterns(x, y, model=modeling.standard,
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

    :param error_func: function used to quantify the quality of the patterns
    (default is the residuals-sum of squared errors).
    :type: function

    :return: For each possible pattern, the error, the splits and the channels
    patterns.
    :rtype: list of tuple
    """
    selected_patterns = []

    reg = _tree.RegressionTree()
    reg = reg.fit(x, y)

    # Models can be symmetric
    channels_patterns = model.channels_patterns()
    channels_patterns += [channels_pattern[::-1]
                          for channels_pattern in channels_patterns]

    for number_of_segments in model.numbers_of_segments():
        prediction_y = reg.predict(x, max_partitions=number_of_segments)
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
    return selected_patterns


def analyze(profile, model=modeling.standard):
    """
    Detect the segments in profile and analyze it.

    By default, it takes the model with the minimal error.

    :param profile: Input profile (containing the X values and the Y values of
    the two channels as column vectors of a matrix).
    :type profile: numpy.ndarray

    :param model: Model defining the patterns to detect and filter (default is
    the standard model defined in dfa.modeling.standard).
    :type model: dfa.modeling.Model

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

    x, y1, y2 = profile[:, 0], profile[:, 1], profile[:, 2]
    y = np.log(y1) - np.log(y2)
    possible_patterns = _select_possible_patterns(x, y, model=model)

    _, splits, channels_pattern = possible_patterns[0]
    splits.insert(0, 0)
    splits.append(x.size-1)
    lengths = np.diff(x[splits])

    return model.search(channels_pattern), lengths


def analyzes(profiles, model=modeling.standard, update_model=True, keys=None):
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
    results (default is None). The tuples must be (experiment name, image name,
    fiber name).
    :type keys: list of tuples

    :return: A data structure containing the detailed measurements.

    :raises ValueError: In case inputs are not valid.
    """
    if type(profiles) != list:
        raise ValueError('Input profiles must be a list of profiles!\n'
                         'It is of type {}...'.format(type(profiles)))

    if any(type(profile) != np.ndarray for profile in profiles):
        raise ValueError('Input profiles must be of type numpy.ndarray!\n'
                         'At least one is not of type numpy.ndarray...')

    if any(profile.shape[0] <= 1 or profile.shape[1] !=3
           for profile in profiles):
        raise ValueError('Input profiles must have a shape equal to Nx3 '
                         '(N>=1 rows and 3 columns)!\n'
                         'At least one does not have this shape...')

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
                             'Index has {} and profiles has {}...'.format(
                len(keys), len(profiles)))

        if any(type(key) != tuple for key in keys):
            raise ValueError('Key index must be of type tuple!\n'
                             'At least one key is not of type tuple...')

        if any(len(key) != 3 for key in keys):
            raise ValueError('Key index must have size equal to 3!\n'
                             'At least one key does not have shape '
                             'equal to 3...')

        index = pd.MultiIndex(levels=[[], [], []], labels=[[], [], []],
                              names=['experiment', 'image', 'fiber'])
    else:
        keys = range(len(profiles))
        index = pd.MultiIndex(levels=[[]], labels=[[]],
                              names=['profile'])

    labels = ['pattern', 'channel', 'length']
    detailed_analysis = pd.DataFrame([], columns=labels, index=index)

    for key, profile in zip(keys, profiles):
        pattern, lengths = analyze(profile, model=model)

        if update_model:
            pattern['freq'] += 1
            pattern['mean'] = [sum_lengths + length
                               for sum_lengths, length
                               in zip(pattern['mean'], lengths)]
            pattern['std'] = [sum_squared_lengths + length**2
                              for sum_squared_lengths, length
                              in zip(pattern['std'], lengths)]

        for length, channel in zip(lengths, pattern['channels']):
            s = pd.Series({labels[0]: pattern['name'],
                           labels[1]: model.channels_names[channel],
                           labels[2]: length},
                          name=key)

            detailed_analysis = detailed_analysis.append(s)

    if update_model:
        for pattern in model.patterns:
            if pattern['freq'] > 0:
                pattern['mean'] = [sum_lengths / pattern['freq']
                                   for sum_lengths in pattern['mean']]
                pattern['std'] = [np.sqrt(sum_squared_lengths / pattern['freq']
                                          - mean_lengths ** 2)
                                  for sum_squared_lengths, mean_lengths
                                  in zip(pattern['std'], pattern['mean'])]

    return detailed_analysis


@removals.remove(removal_version='?')
def segments(profile):
    """
    Detect the segments in profile.

    :param profile: Input profile (containing the X values and the two Y values
    of channels).
    :type profile: A matrix (numpy.ndarray of size (N,3)).

    :return: Segments' properties (successive channel domination and change
    points).
    :rtype: A list of successive channel domination and a list of change points.

    :raises ValueError: In case the input value is not a ndarray of shape Nx3.
    """
    if type(profile) != np.ndarray:
        raise ValueError('Input profile should be a numpy ndarray!')

    if profile.shape[0] <= 1 or profile.shape[1] != 3:
        raise ValueError('Input profile should have a shape as Nx3 '
                         '(N rows and 3 columns)!')

    # Find the best regression tree model on the log ratio of first channel
    # over second channel
    profile_x = profile[:, 0].reshape(profile[:, 0].size, 1)
    profile_y = np.log(profile[:, 1] / profile[:, 2]).reshape(profile_x.shape)
    prediction_y, change_indices, _ = _choose_piecewise_model(profile_x,
                                                              profile_y)

    # Prepare output
    change_points = profile_x[change_indices].astype(float).ravel().tolist()
    change_points.insert(0, float(profile_x[0][0]))
    change_points.insert(len(change_points),
                         float(profile_x[profile_x.size-1][0]))

    change_indices.insert(len(change_indices), profile_x.size - 1)
    channels = [0 if prediction_y[change_index] > 0 else 1
                for change_index in change_indices]

    # begin post-conditions
    assert type(change_points) == list
    assert all(type(change_point) == float for change_point in change_points)
    assert type(channels) == list
    assert all(channel == 0 or channel == 1 for channel in channels)
    assert len(channels) == len(change_points) - 1
    # end post-conditions

    return channels, change_points


@removals.remove(removal_version='?')
def quantify(channels, change_points, patterns):
    """
    Quantify the segments detected with the segments function.

    :param channels: Successive channels (segment membership).
    :type channels: List of 0 or 1.

    :param change_points: Change points (inter-segments).
    :type change_points: List of floating point numbers.

    :param patterns: A dictionary of patterns to detect.
    :type patterns: A dictionary which associate a binary list (0 and 1) to
    pattern's name.

    :return: Detected pattern (with dictionary) and lengths.
    :rtype: A string and a list of lengths.
    """
    if type(change_points) != list \
            or any(type(change_point) != float
                   for change_point in change_points):
        raise ValueError('The change points must be a list of floating '
                         'points numbers!')

    if type(channels) != list \
            or any(channel != 0 and channel != 1 for channel in channels):
        raise ValueError('The channels must be a list of 0 or 1!')

    if len(channels) != len(change_points) - 1:
        raise ValueError('There must be one more channel in list than '
                         'change point!')

    if type(patterns) != dict:
        raise ValueError('The input patterns must be a dictionary!')

    if any(type(item[0]) != tuple or type(item[1]) != str
           for item in patterns.items()):
        raise ValueError('The input patterns keys/values must be a '
                         'tuple/string!')

    if any(type(elem) != int for item in patterns.items() for elem in item[0]):
        raise ValueError('The input patterns keys must be integers!')

    if any(elem != 0 and elem != 1
           for item in patterns.items()
           for elem in item[0]):
        raise ValueError('The input patterns keys must contains only 0 and 1!')

    pattern_name = patterns.get(tuple(channels), 'NA')
    lengths = np.diff(change_points).astype(float).ravel().tolist()

    # begin post-conditions
    assert type(pattern_name) == str
    # assert pattern_name in patterns.values()
    assert type(lengths) == list
    assert all(type(length) == float for length in lengths)
    assert len(lengths) == len(channels)
    # end post-conditions

    return pattern_name, lengths


if __name__ == '__main__':
    import os
    import copy

    # Read profiles
    path = '../data/profiles'
    file_names = os.listdir(path)
    file_names = [file_name for file_name in file_names
                  if file_name.endswith('.csv')]
    profiles = [np.loadtxt('{}/{}'.format(path, file_name), delimiter=',',
                           skiprows=1, usecols=(0, 1, 2))
                for file_name in file_names]

    experiments = [file_name.split('_')[0] for file_name in file_names]
    images = [file_name.split('_')[1] for file_name in file_names]
    fibers = [file_name.split('.')[0].split('_')[2] for file_name in file_names]

    # Find patterns and save quantification to csv files
    model = copy.deepcopy(modeling.standard)
    model.initialize_for_quantification()
    detailed_analysis = analyzes(profiles,
                                 model=model,
                                 keys=list(zip(experiments, images, fibers)))
    print(detailed_analysis)
    model._normalize_frequencies()
    print(model.patterns)

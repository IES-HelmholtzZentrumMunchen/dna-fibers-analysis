import numpy as np
from sklearn.tree import DecisionTreeRegressor


class BinaryNode:
    def __init__(self, values=None, left=None, right=None):
        """
        Constructor of Binary nodes.

        :param values: Values associated with the current node.
        :type values: any

        :param left: Left child (subtree, node or None if the current node is
        a leaf).
        :type left: BinaryNode

        :param right: Right child (subtree, node or None if the current node is
        a leaf).
        :type right: BinaryNode
        """
        self.values = values
        self.left = left
        self.right = right

    def leaves(self):
        """
        Get the leaves of the tree as a list.

        :return: Leaves of the current tree.
        :rtype: list of BinaryNodes
        """
        def _recursive_leaves_search(tree):
            if tree.left is None or tree.right is None:
                return [tree]
            else:
                return _recursive_leaves_search(tree.left) + \
                       _recursive_leaves_search(tree.right)

        return _recursive_leaves_search(self)


class RegressionTree:
    def __init__(self, max_depth=3, min_samples=20):
        """
        Constructor of regression tree.

        :param max_depth: Maximum depth of the binary tree (default is 3).
        :type max_depth: positive int

        :param min_samples: Minimum number of samples per leaves (default
        is 20).
        :type min_samples: strictly positive int
        """
        self.max_depth = max_depth
        self.min_samples = min_samples

        self._tree = None

    def fit(self, x, y):
        """
        Compute the binary regression tree.

        The computation is performed using a recursive scheme. The split
        iteration
        is performed using a fast algorithm with linear complexity in the
        number of
        points.

        :param x: Input independent variables.
        :type x: numpy.ndarray (1D)

        :param y: Input dependent variables.
        :type y: numpy.ndarray (1D)

        :return: The regression tree object.
        """
        min_samples_for_split = 2 * self.min_samples

        def _fast_optimal_binary_split(y):
            """
            Split into binary partition using fast algorithm (linear
            complexity in
            the number of points).

            The error is the opposite of the weighted sum of square values.
            """

            def _calculate_error():
                return - (s1 ** 2 / n1 + s2 ** 2 / n2)

            s1 = y[:self.min_samples].sum()
            n1 = y[:self.min_samples].size
            s2 = y[self.min_samples:].sum()
            n2 = y.size - n1

            optimal_k = 0
            optimal_error = _calculate_error()

            for k in range(1, y.size - self.min_samples):
                s1 += y[k]
                n1 += 1
                s2 -= y[k]
                n2 -= 1

                error = _calculate_error()

                if error < optimal_error:
                    optimal_k = k
                    optimal_error = error

            return optimal_k, optimal_error

        def _regression_tree_recursion(x, y, max_depth):
            if max_depth == 1:
                k, _ = _fast_optimal_binary_split(y)

                return BinaryNode(values=(y.mean(), x.min(), x.max()),
                                  left=BinaryNode(
                                      values=(y[:k].mean(),
                                              x[:k].min(),
                                              x[:k].max())),
                                  right=BinaryNode(
                                      values=(y[k:].mean(),
                                              x[k:].min(),
                                              x[k:].max())))
            else:
                k, _ = _fast_optimal_binary_split(y)

                y_left, x_left = y[:k], x[:k]
                y_right, x_right = y[k:], x[k:]
                subtree_left = None
                subtree_right = None

                if y_left.size >= min_samples_for_split and \
                   y_right.size >= min_samples_for_split:
                    subtree_left = _regression_tree_recursion(x_left, y_left,
                                                              max_depth - 1)
                    subtree_right = _regression_tree_recursion(x_right, y_right,
                                                               max_depth - 1)

                return BinaryNode(values=(y.mean(), x.min(), x.max()),
                                  left=subtree_left,
                                  right=subtree_right)

        if self.max_depth == 0:
            self._tree = BinaryNode(values=(y.mean(), x.min(), x.max()))
        else:
            self._tree = _regression_tree_recursion(x, y, self.max_depth)

        return self

    def predict(self, x):
        """
        Predict dependent values with the previously computed binary tree.

        :param x: Input independent variables.
        :type x: numpy.ndarray (1D)

        :return: Estimated dependent variables.
        :rtype: numpy.ndarray (1D)
        """
        leaves = self._tree.leaves()
        y = np.zeros(x.shape)

        for leaf in leaves:
            y[np.bitwise_and(x >= leaf.values[1],
                             x <= leaf.values[2])] = leaf.values[0]

        return y


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
    from matplotlib import pyplot as plt

    # Read profiles
    path = '../data/profiles'
    file_names = os.listdir(path)
    file_names = [file_name for file_name in file_names
                  if file_name.endswith('.csv')]
    profiles = [np.loadtxt('{}/{}'.format(path, file_name), delimiter=',',
                           skiprows=1, usecols=(0, 1, 2))
                for file_name in file_names]

    # Find regression tree and display it
    columns = 3
    fig = plt.figure()

    for index, profile in enumerate(profiles):
        profile_x = profile[:, 0].reshape(profile[:, 0].size, 1)
        profile_y = np.log(profile[:, 1]/profile[:, 2]).reshape(profile_x.shape)

        prediction_y, points, error = _choose_piecewise_model(profile_x,
                                                              profile_y)

        ax = fig.add_subplot(np.ceil(len(profiles)/columns), columns, index+1)
        ax.scatter(profile[:, 0], profile_y, c='black')
        ax.plot(profile_x, prediction_y, '-r')

        for point in points:
            ax.plot(profile_x[[point, point]], [min(profile_y),
                                                max(profile_y)], '--g')

        ax.set_title(file_names[index].split('.')[0].replace('_', '-'))

    plt.tight_layout()
    plt.show()

    # Detect segments and quantify
    patterns = {
        (0, 1, 0): 'Origin',
        (0, 1): 'Ongoing fork',
        (1, 0): 'Ongoing fork',
        (1, 0, 1): 'Termination'
    }

    for profile, file_name in zip(profiles, file_names):
        print(file_name)
        channels, change_points = segments(profile)
        print(channels, change_points)
        pattern_name, lengths = quantify(channels, change_points, patterns)
        print(pattern_name, lengths)
        print()

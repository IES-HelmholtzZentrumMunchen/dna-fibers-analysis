"""
Module for regression trees management.

In particular, it contains a class for binary trees and a class for regression
trees.
"""
import numpy as np


class BinaryNode:
    """
    Defines a binary tree with a single class.

    This class defines a general node. A leaf is a node without any children.
    """
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
        self.left = left
        self.right = right

        try:
            self.values = tuple(values)
        except TypeError:
            self.values = tuple([values])

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

    def depth_first(self):
        """
        Go trough the tree with depth-first strategy.

        :return: A generator to the nodes.
        :rtype: generator
        """
        def _recursive_depth_first(node):
            if node is not None:
                yield node
                yield from _recursive_depth_first(node.left)
                yield from _recursive_depth_first(node.right)

        return _recursive_depth_first(self)

    def display(self, offset_factor=2, values_to_display=slice(None)):
        """
        Display the tree in a terminal.

        :param offset_factor: Width tabulation factor.
        :type offset_factor: strictly positive int

        :param values_to_display: Slicing object used to select values to
        display.
        :type values_to_display: slice
        """
        def _recursive_display(tree, offset, offset_factor):
            if tree is not None:
                print('{:->{offset}}{}'.format('',
                                               tree.values[values_to_display],
                                               offset=offset_factor * offset))
                _recursive_display(tree.left, offset + 1, offset_factor)
                _recursive_display(tree.right, offset + 1, offset_factor)

        _recursive_display(self, 0, offset_factor)

    def print(self, filename, values_to_print=slice(None), out='dot'):
        """
        Write binary tree to a DOT file.

        :param filename: Output filename to write to.
        :type filename: path (str)

        :param values_to_print: Slicing object used to select values to print.
        :type values_to_print: slice

        :param out: Tell the final destination of the print to choose the
        separator between the node id and the values.
        :type out: str
        """
        if out == 'latex':
            sep = '\\\\\\\\'
        elif out == 'dot':
            sep = '\\n'
        else:
            sep = ' '

        def _recursive_dot(tree, id_name=0):
            if tree.left is None or tree.right is None:
                return []
            else:
                output = [
                    '"{}{}{}" -> "{}{}{}";'.format(
                        id_name, sep, tree.values[values_to_print],
                        10*id_name+1, sep, tree.left.values[values_to_print]),
                    '"{}{}{}" -> "{}{}{}";'.format(
                        id_name, sep, tree.values[values_to_print],
                        10*id_name+2, sep, tree.right.values[values_to_print])
                ]

                return (output + _recursive_dot(tree.left, 10*id_name+1) +
                        _recursive_dot(tree.right, 10*id_name+2))

        lines = _recursive_dot(self)
        lines.insert(0, 'digraph G {')
        lines.append('}')

        with open(filename, 'w') as graph_file:
            for line in lines:
                graph_file.write(line + '\n')


class RegressionTree:
    """
    Defines a regression tree for 1D dependent variables.
    """
    def __init__(self, max_depth=3, min_samples=1):
        """
        Constructor of regression tree.

        :param max_depth: Maximum depth of the binary tree (default is 3).
        :type max_depth: positive int

        :param min_samples: Minimum number of samples per leaves (default
        is 1).
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
            n1 = self.min_samples
            s2 = y[self.min_samples:].sum()
            n2 = y.size - n1

            optimal_k = self.min_samples - 1
            optimal_error = _calculate_error()

            for k in range(self.min_samples - 1, y.size - self.min_samples - 1):
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

                return BinaryNode(values=(y.mean(), x.min(), x.max(),
                                          np.power(y-y.mean(), 2).sum()),
                                  left=BinaryNode(
                                      values=(y[:k].mean(),
                                              x[:k].min(),
                                              x[:k].max(),
                                              np.power(y[:k]-y[:k].mean(),
                                                       2).sum())),
                                  right=BinaryNode(
                                      values=(y[k:].mean(),
                                              x[k:].min(),
                                              x[k:].max(),
                                              np.power(y[k:]-y[k:].mean(),
                                                       2).sum())))
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

                return BinaryNode(values=(y.mean(), x.min(), x.max(),
                                          np.power(y-y.mean(), 2).sum()),
                                  left=subtree_left,
                                  right=subtree_right)

        if self.max_depth == 0:
            self._tree = BinaryNode(values=(y.mean(), x.min(), x.max(),
                                            np.power(y-y.mean(), 2).sum()))
        else:
            self._tree = _regression_tree_recursion(x, y, self.max_depth)

        return self

    def find_partitions(self):
        """
        Find the possible partition of the space and order them by a criterion.

        Find partitions of the independent variables from the fitted tree with
        a best-first approach. The function maximized is the variance percentage
        improvement per split.
        """
        nodes_to_visit = [self._tree]

        # TODO implement a best-first search to find partitions
        # for node in nodes_to_visit:
        #     1 - (node.left.values[3] + node.right.values[3]) / node.values[3]

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

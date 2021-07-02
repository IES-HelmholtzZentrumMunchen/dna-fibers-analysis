"""
Module for regression trees management.

In particular, it contains a class for binary trees and a class for regression
trees.
"""
import numpy as np


class BinaryNode:
    """Defines a binary tree with a single class.

    This class defines a general node. A leaf is a node without any children.

    Attributes
    ----------
    parent : BinaryNode
        Parent of the node.

    left : BinaryNode
        Left subtree.

    right : BinaryNode
        Right subtree.

    values : any
        Values attached to the node.
    """
    def __init__(self, values=None, left=None, right=None):
        """Constructor of Binary nodes.

        Parameters
        ----------
        values : None | List[T]
            Values associated with the current node.

        left : BinaryNode
            Left child (subtree, node or None if the current node is a leaf).

        right : BinaryNode
            Right child (subtree, node or None if the current node is a leaf).
        """
        self.parent = None
        self.left = left
        self.right = right

        if self.left is not None:
            self.left.parent = self

        if self.right is not None:
            self.right.parent = self

        try:
            self.values = tuple(values)
        except TypeError:
            self.values = tuple([values])

    def leaves(self):
        """Get the leaves of the tree as a list.

        Returns
        -------
        list of BinaryNodes
            Leaves of the current tree.
        """
        def _recursive_leaves_search(tree):
            if tree.left is None or tree.right is None:
                return [tree]
            else:
                return _recursive_leaves_search(tree.left) + \
                       _recursive_leaves_search(tree.right)

        return _recursive_leaves_search(self)

    def depth_first(self):
        """Traverse the tree with depth-first strategy.

        Returns
        -------
        generator
            A generator to the nodes in depth-first order.
        """
        def _recursive_depth_first(node):
            if node is not None:
                yield node
                yield from _recursive_depth_first(node.left)
                yield from _recursive_depth_first(node.right)

        return _recursive_depth_first(self)

    def breadth_first(self):
        """Traverse the tree with breadth-first strategy.

        Returns
        -------
        generator
            A generator to the nodes in breadth-first order.
        """
        nodes_to_visit = [self]

        while nodes_to_visit:
            node = nodes_to_visit.pop(0)

            if node is not None:
                yield node
                nodes_to_visit += [node.left, node.right]

    def best_first(self, func):
        """Traverse the tree with best-first strategy.

        The nodes are ordered using the best-first strategy but keeping the
        hierarchy of the tree. For instance, a node cannot be visited if its
        parent is hast not been already visited, even if it has a greater
        output function value than his parent.

        Parameters
        ----------
        func : callable function
            Function of the values for choosing best node.

        Returns
        -------
        generator
            A generator to the nodes in best-first order.
        """
        nodes_to_visit = [(self, func(self))]

        while nodes_to_visit:
            nodes_to_visit.sort(key=lambda e: e[1])
            node, _ = nodes_to_visit.pop(0)

            yield node

            if node.left is not None:
                nodes_to_visit.append((node.left, func(node.left)))

            if node.right is not None:
                nodes_to_visit.append((node.right, func(node.right)))

    def display(self, offset_factor=2, values_to_display=None):
        """Display the tree in a terminal.

        Parameters
        ----------
        offset_factor : strictly positive int
            Width tabulation factor.

        values_to_display : slice
            Slicing object used to select values to display.
        """
        if values_to_display is None:
            values_to_display = slice(len(self.values))

        def _recursive_display(tree, offset, offset_factor):
            if tree is not None:
                print('{:->{offset}}{}'.format('',
                                               tree.values[values_to_display],
                                               offset=offset_factor * offset))
                _recursive_display(tree.left, offset + 1, offset_factor)
                _recursive_display(tree.right, offset + 1, offset_factor)

        _recursive_display(self, 0, offset_factor)

    def print(self, filename, values_to_print=None, out='dot'):
        """Write binary tree to a DOT file.

        Parameters
        ----------
        filename : path (str)
            Output filename to write to.

        values_to_print : slice
            Slicing object used to select values to print.

        out : str
            Tell the final destination of the print to choose the separator
            between the node id and the values.
        """
        if values_to_print is None:
            values_to_print = slice(len(self.values))

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
    """Defines a regression tree for 1D dependent variables.

    Building a general tree being NP-hard problem, we use instead a heuristic
    by constructing a binary tree (CART algorithm).

    Each node of the binary tree represents a particular split.

    Attributes
    ----------
    max_depth : strictly positive int
            Maximum depth of the binary tree (default is 3).

    min_samples : strictly positive int
        Minimum number of samples per leaves (default is 2).

    _tree : BinaryNode
        Tree to be estimated by regression.
    """
    def __init__(self, max_depth=3, min_samples=2):
        """Constructor of regression tree.

        Parameters
        ----------
        max_depth : strictly positive int
            Maximum depth of the binary tree (default is 3).

        min_samples : strictly positive int
            Minimum number of samples per leaves (default is 2).
        """
        self.max_depth = max_depth
        self.min_samples = min_samples

        self._tree = None

    def fit(self, x, y, error_func=lambda y: np.power(y-y.mean(), 2).mean()):
        """Compute the binary regression tree.

        The computation is performed using a recursive scheme. The split
        iteration is performed using a fast algorithm with linear complexity
        in the number of points.

        Parameters
        ----------
        x : numpy.ndarray (1D)
            Input independent variables.

        y : numpy.ndarray (1D)
            Input dependent variables.

        error_func : lambda function
            Error function to use (default is SSE).

        Returns
        -------
        RegressionTree
            The regression tree object.
        """
        min_samples_for_split = self.min_samples * (self.min_samples - 1)

        def _fast_optimal_binary_split(y):
            """Split into binary partition using fast algorithm (linear
            complexity in the number of points).

            The error is the opposite of the weighted sum of square values.
            """
            def _calculate_error():
                return - (s1 ** 2 / n1 + s2 ** 2 / n2)

            s1 = y[:self.min_samples].sum()
            n1 = self.min_samples
            s2 = y[self.min_samples:].sum()
            n2 = y.size - n1

            optimal_k = self.min_samples
            optimal_error = _calculate_error()

            for k in range(self.min_samples, y.size - self.min_samples - 1):
                s1 += y[k]
                n1 += 1
                s2 -= y[k]
                n2 -= 1

                error = _calculate_error()

                if error < optimal_error:
                    optimal_k = k
                    optimal_error = error

            return optimal_k

        def _regression_tree_recursion(x, y, max_depth):
            if max_depth == 1:
                k = _fast_optimal_binary_split(y)

                y_left, y_right = y[:k], y[k:]

                return BinaryNode(values=(x.min(), x[k], x.max(),
                                          y_left.mean(), y_right.mean(),
                                          error_func(y_left),
                                          error_func(y_right)))
            else:
                k = _fast_optimal_binary_split(y)

                y_left, x_left = y[:k], x[:k]
                y_right, x_right = y[k:], x[k:]
                subtree_left = None
                subtree_right = None

                if y_left.size > min_samples_for_split:
                    subtree_left = _regression_tree_recursion(x_left, y_left,
                                                              max_depth - 1)

                if y_right.size > min_samples_for_split:
                    subtree_right = _regression_tree_recursion(x_right, y_right,
                                                               max_depth - 1)

                return BinaryNode(values=(x.min(), x[k], x.max(),
                                          y_left.mean(), y_right.mean(),
                                          error_func(y_left),
                                          error_func(y_right)),
                                  left=subtree_left,
                                  right=subtree_right)

        if y.size >= min_samples_for_split:
            self._tree = _regression_tree_recursion(x, y,
                                                    self.max_depth)

        # Since we use the node is split strategy, we also
        # add artificially a root node at the top, representing 0 split.
        self._tree = BinaryNode(values=(x.min(), x.max(), x.max(),
                                        y.mean(), y.mean(),
                                        error_func(y), error_func(y)),
                                left=self._tree)

        return self

    def _partitioning_nodes(self, max_partitions=None,
                            constraint_func=lambda _: True):
        """Find the possible partitioning nodes with specified maximal segments.

        Find partitions of the independent variables from the fitted tree with
        a best-first approach. The function try to minimize the normalized
        residuals per split and per branch.

        Parameters
        ----------
        max_partitions : int greater than 0 or None
            The maximum number of partitions to find (default is None).

        constraint_func : callable function
            Function that check a constraint. Default is no constraint (the
            function returns always True).

        Returns
        -------
        list of BinaryNode
            The splitting nodes in the order of their traversal.
        """
        def _error_function(node):
            """This is an heuristic function to help finding the best next node.

            It sums the error of the current split and normalize it by the
            error coming from the corresponding branch. The lower is this ratio,
            the better is the "importance" of the current split node.
            """
            constraint = constraint_func(node)
            split_error = sum(node.values[5:7])

            if node is node.parent.left:
                left_error = split_error / node.parent.values[5]
                return left_error if constraint else left_error + 1000000
            else:
                right_error = split_error / node.parent.values[6]
                return right_error if constraint else right_error + 1000000

        if max_partitions is None:
            max_partitions = 2**self.max_depth

        bests = self._tree.left.best_first(func=_error_function)
        nodes = [self._tree]

        try:
            for _ in range(max_partitions - 1):
                nodes.append(next(bests))
        except StopIteration:
            pass

        return nodes

    def predict(self, x, max_partitions=None, constraint_func=lambda _: True):
        """Predict dependent values with the previously computed binary tree.

        Parameters
        ----------
        x : numpy.ndarray (1D)
            Input independent variables.

        max_partitions : int greater than 0 or None
            The maximum number of partitions to find (default is None).

        constraint_func : function
            Function that check a constraint. Default is no constraint (the
            function returns always True).

        Returns
        -------
        numpy.ndarray
            The predicted dependent values.
        """
        y = np.zeros(x.shape)

        for node in self._partitioning_nodes(max_partitions=max_partitions,
                                             constraint_func=constraint_func):
            y[np.bitwise_and(
                node.values[0] <= x,
                x <= node.values[1])] = node.values[3]
            y[np.bitwise_and(
                node.values[1] <= x,
                x <= node.values[2])] = node.values[4]

        return y

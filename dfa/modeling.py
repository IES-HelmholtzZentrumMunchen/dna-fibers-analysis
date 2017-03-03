"""
Model management of the DNA fiber analysis package.

Use this module to use the default model, create custom models and use the
convenience tools for models management.
"""
import numpy as np


class Model:
    """
    Pattern model management.

    This is a convenience class used for model management. Once the model of
    patterns is set (see below), it can be saved and patterns can be simulated
    randomly from it. It can also be loaded from file.

    A model is a list of pattern structures. Pattern structure is a dictionary
    with the following keys:
    - 'name': the name of the pattern (ex: 'ongoing fork')
    - 'freq': its frequency of appearance (ex: 0.7)
    - 'channels': its actual pattern of channels (ex: [0, 1])
    - 'mean': the mean length of branches (ex: [100, 90])
    - 'std': the standard deviation of branches (ex: [10, 5])
    """
    def __init__(self, patterns):
        """
        Initialize the model with the specified patterns.

        :param patterns: List of patterns.
        :type patterns: list of dict
        """
        self.patterns = patterns

        self._update_frequencies()
        self._normalize_frequencies()

    def _update_frequencies(self):
        """
        Update the class shortcut for patterns frequencies.
        """
        self._frequencies = [pattern['freq'] for pattern in self.patterns]

    def _normalize_frequencies(self):
        """
        Normalize the patterns frequencies.
        """
        norm_factor = 1.0 / sum(self._frequencies)

        for pattern in self.patterns:
            pattern['freq'] *= norm_factor

        self._update_frequencies()

    def numbers_of_segments(self):
        """
        Get the number of segments defined by the modeling.

        :return: A list of all possible segment numbers.
         :rtype: list of strictly positive int
        """
        numbers_of_segments = []

        for pattern in self.patterns:
            if len(pattern['channels']) not in numbers_of_segments:
                numbers_of_segments.append(len(pattern['channels']))

        return numbers_of_segments

    def channels_patterns(self):
        """
        Get the channels patterns as a list.

        :return: A list of all channels patterns in the current model.
        :rtype: list of list of int
        """
        return [pattern['channels'] for pattern in self.patterns]

    def save(self, filename):
        print(filename, self.patterns)
        raise RuntimeError('Not yet implemented!')

    @staticmethod
    def load(filename):
        raise RuntimeError('Not yet implemented!')

    def simulate_patterns(self, number):
        """
        Simulate number patterns.

        :param number: Number of patterns to simulate.
        :type number: int

        :return: The channels (the patterns branch) and the lengths (the
        branches lengths).
        :rtype:list of list of int and list of list of float
        """
        patterns = np.random.choice(self.patterns, number, p=self._frequencies)

        channels_pattern, lengths = [], []

        for pattern in patterns:
            channels_pattern.append(pattern['channels'])

            lengths.append([std * np.random.randn() + mean
                            for mean, std in
                            zip(pattern['mean'], pattern['std'])])

        return channels_pattern, lengths


standard = Model([
    # Take results carefully for 1-segments patterns, it might not be reliable!
    {'name': 'stalled',
     'freq': 0.6,
     'channels': [0],
     'mean': [100],
     'std': [10]},
    {'name': 'ongoing fork',
     'freq': 0.6,
     'channels': [0, 1],
     'mean': [100, 90],
     'std': [10, 5]},
    {'name': '1st label origin',
     'freq': 0.3,
     'channels': [0, 1, 0],
     'mean': [75, 150, 90],
     'std': [10, 30, 20]},
    {'name': '2nd label termination',
     'freq': 0.1,
     'channels': [1, 0, 1],
     'mean': [100, 50, 100],
     'std': [20, 5, 25]},
])

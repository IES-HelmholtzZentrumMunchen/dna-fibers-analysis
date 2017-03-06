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
    1) keys for simulation
    - 'name': the name of the pattern (ex: 'ongoing fork')
    - 'freq': its frequency of appearance (ex: 0.7)
    - 'channels': its actual pattern of channels (ex: [0, 1])
    - 'mean': the mean length of branches (ex: [100, 90])
    - 'std': the standard deviation of branches (ex: [10, 5])
    2) keys for quantification
    - 'count': the number of sample belonging to the pattern (ex: 11)
    - 'lengths': the sample lengths for each of the segment
    """
    def __init__(self, patterns, channels_names=None):
        """
        Initialize the model with the specified patterns.

        :param patterns: List of patterns.
        :type patterns: list of dict

        :param channels_names: List of names of channels in the same order as in
        patterns (0, 1, etc.).
        :type channels_names: list
        """
        self.patterns = patterns
        self.channels_names = channels_names

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
        self._update_frequencies()

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

    def search(self, channels_pattern):
        """
        Get the pattern corresponding to the input channels pattern.

        The symmetric channels patterns are also taken into account.

        :param channels_pattern: Input sequence of channels.
        :type channels_pattern: list of int

        :return: The pattern.
        :rtype: dict or None (if no pattern is found)
        """
        for pattern in self.patterns:
            if pattern['channels'] == channels_pattern:
                return pattern

        return None

    @staticmethod
    def append_sample(pattern, lengths):
        """
        Append a sample to the specified pattern.

        This is a convenience method for updating the model (count and lengths)
        with a new sample.

        :param pattern: Specified pattern to update (it must be an element
        of self.patterns as no check will be performed).
        :type pattern: dict as defined to be a pattern

        :param lengths: Lengths of the sample.
        :type lengths: list of float
        """
        pattern['count'] += 1

        for length, samples in zip(lengths, pattern['lengths']):
            samples.append(length)

    def update_model(self):
        """
        Update the model after appending samples to patterns.

        This is a convenience method for updating the model. It can be called
        after all samples have been appended with method Model.append_sample to
        be used for simulation for instance.
        """
        for pattern in self.patterns:
            if pattern['count'] > 0:
                pattern['mean'] = [sum(samples) / pattern['count']
                                   for samples in pattern['lengths']]

                pattern['std'] = [np.sqrt(sum([sample**2 for sample in samples])
                                  / pattern['count'] - mean**2)
                                  for samples, mean in zip(pattern['lengths'],
                                                           pattern['mean'])]
            else:
                pattern['mean'] = [0 for _ in pattern['mean']]
                pattern['std'] = [0 for _ in pattern['std']]

        self._normalize_frequencies()

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
     'std': [10],
     'count': 0,
     'lengths': [[]]},
    {'name': 'ongoing fork',
     'freq': 0.6,
     'channels': [0, 1],
     'mean': [100, 90],
     'std': [10, 5],
     'count': 0,
     'lengths': [[], []]},
    {'name': '1st label origin',
     'freq': 0.3,
     'channels': [0, 1, 0],
     'mean': [75, 150, 90],
     'std': [10, 30, 20],
     'count': 0,
     'lengths': [[], [], []]},
    {'name': '2nd label termination',
     'freq': 0.1,
     'channels': [1, 0, 1],
     'mean': [100, 50, 100],
     'std': [20, 5, 25],
     'count': 0,
     'lengths': [[], [], []]}], channels_names=['CIdU', 'IdU'])

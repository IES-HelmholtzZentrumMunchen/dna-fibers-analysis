"""
Model management of the DNA fiber analysis package.

Use this module to use the default model, create custom models and use the
convenience tools for models management.
"""
import numpy as np


class Model:
    """Pattern model management.

    This is a convenience class used for model management. Once the model of
    patterns is set (see below), it can be saved and patterns can be simulated
    randomly from it. It can also be loaded from file.

    A model is a list of pattern structures. Pattern structure is a dictionary
    with the following keys:
    1. keys for simulation

       - 'name': the name of the pattern (ex: 'ongoing fork')
       - 'freq': its frequency of appearance (ex: 0.7)
       - 'channels': its actual pattern of channels (ex: [0, 1])
       - 'mean': the mean length of branches (ex: [100, 90])
       - 'std': the standard deviation of branches (ex: [10, 5])

    2. keys for quantification

      - 'count': the number of sample belonging to the pattern (ex: 11)

    The keys for simulation can be filled directly or can be updated after a
    quantification.
    """
    def __init__(self, patterns, channels_names=None):
        """Initialize the model with the specified patterns.

        Parameters
        ----------
        patterns : list of dict
            List of patterns.

        channels_names : List[str]
            List of names of channels in the same order as in patterns
            (0, 1, etc.).
        """
        self.patterns = patterns
        self.channels_names = channels_names

        self._normalize_frequencies()

    def _update_frequencies(self):
        """
        Update the class shortcut for patterns frequencies.
        """
        for pattern in self.patterns:
            pattern['freq'] = pattern['count']

    def _normalize_frequencies(self):
        """
        Normalize the patterns frequencies.
        """
        try:
            norm_factor = (1.0 / sum(pattern['freq'] for pattern in self.patterns))
        except ZeroDivisionError:
            norm_factor = 0

        for pattern in self.patterns:
            pattern['freq'] *= norm_factor

    def numbers_of_segments(self):
        """Get the number of segments defined by the modeling.

        Returns
        -------
        list of strictly positive int
            A list of all possible segment numbers.
        """
        numbers_of_segments = []

        for pattern in self.patterns:
            if len(pattern['channels']) not in numbers_of_segments:
                numbers_of_segments.append(len(pattern['channels']))

        return numbers_of_segments

    def channels_patterns(self):
        """Get the channels patterns as a list.

        Returns
        -------
        list of list of int
            A list of all channels patterns in the current model.
        """
        return [pattern['channels'] for pattern in self.patterns]

    def search(self, channels_pattern):
        """Get the pattern corresponding to the input channels pattern.

        Parameters
        ----------
        channels_pattern : list of int
            Input sequence of channels.

        Returns
        -------
        dict or None (if no pattern is found)
            The pattern.
        """
        for pattern in self.patterns:
            if pattern['channels'] == channels_pattern:
                return pattern

        return None

    def initialize_model(self):
        """
        Initialize the model for preparing quantification.
        """
        for pattern in self.patterns:
            pattern['count'] = 0
            pattern['mean'] = [0 for _ in pattern['mean']]
            pattern['std'] = [0 for _ in pattern['mean']]

    @staticmethod
    def append_sample(pattern, lengths):
        """Append a sample to the specified pattern.

        This is a convenience method for updating the model (count and lengths)
        with a new sample.

        Parameters
        ----------
        pattern : dict as defined to be a pattern
            Specified pattern to update (it must be an element of self.patterns
            as no check will be performed).

        lengths : list of float
            Lengths of the sample.
        """
        pattern['count'] += 1

        pattern['mean'] = [sum_lengths+length
                           for sum_lengths, length
                           in zip(pattern['mean'], lengths)]
        pattern['std'] = [sum_squared_lengths + length**2
                          for sum_squared_lengths, length
                          in zip(pattern['std'], lengths)]

    def update_model(self):
        """Update the model after appending samples to patterns.

        This is a convenience method for updating the model. It can be called
        after all samples have been appended with method Model.append_sample to
        be used for simulation for instance.
        """
        for pattern in self.patterns:
            if pattern['count'] > 0:
                pattern['mean'] = [sum_lengths / pattern['count']
                                   for sum_lengths in pattern['mean']]

                pattern['std'] = [np.sqrt(sum_squared_lengths
                                          / pattern['count'] - mean**2)
                                  for sum_squared_lengths, mean
                                  in zip(pattern['std'], pattern['mean'])]
            else:
                pattern['mean'] = [0 for _ in pattern['mean']]
                pattern['std'] = [0 for _ in pattern['std']]

        self._update_frequencies()
        self._normalize_frequencies()

    def save(self, filename):
        """Save the model description of patterns to a text file.

        Parameters
        ----------
        filename : str
            Path where to save the model.
        """
        with open(filename, 'w') as file:
            file.write(str(self.patterns))
            file.write('\n')
            file.write(str(self.channels_names))

    @staticmethod
    def load(filename):
        """Load a model from its description in a text file.

        filename : str
            Path where the model description to load is.
        """
        with open(filename, 'r') as file:
            patterns = eval(file.readline())
            channels_names = eval(file.readline())

        return Model(patterns=patterns, channels_names=channels_names)

    def print(self):
        """
        Print the model on the standard output.
        """
        for pattern in self.patterns:
            print('\n{}: {}% ({} samples)'.format(
                pattern['name'], 100*pattern['freq'], pattern['count']))
            print('pattern: {}, lengths mean: {}, lengths std: {}'.format(
                [self.channels_names[channel]
                 for channel in pattern['channels']],
                pattern['mean'], pattern['std']))

    def simulate_patterns(self, number):
        """Simulate number patterns.

        Parameters
        ----------
        number : int
            Number of patterns to simulate.

        Returns
        -------
        list of list of int and list of list of float
            The channels (the patterns branch) and the lengths (the branches
            lengths).
        """
        patterns = np.random.choice(self.patterns, number,
                                    p=[pattern['freq']
                                       for pattern in self.patterns])

        channels_pattern, lengths = [], []

        for pattern in patterns:
            channels_pattern.append(pattern['channels'])

            lengths.append([std * np.random.randn() + mean
                            for mean, std in
                            zip(pattern['mean'], pattern['std'])])

        return channels_pattern, lengths


standard = Model([
    # Take results carefully for 1-segments patterns, it might not be reliable!
    {'name': 'stalled fork/1st label termination',
     'freq': 1,
     'channels': [1],
     'mean': [100],
     'std': [10],
     'count': 0},
    {'name': '2nd label termination',
     'freq': 1,
     'channels': [1, 0, 1],
     'mean': [100, 100, 100],
     'std': [10, 10, 10],
     'count': 0},
    {'name': 'ongoing fork',
     'freq': 1,
     'channels': [1, 0],
     'mean': [100, 90],
     'std': [10, 5],
     'count': 0},
    {'name': '1st label origin',
     'freq': 1,
     'channels': [0, 1, 0],
     'mean': [75, 150, 90],
     'std': [10, 30, 20],
     'count': 0},
    {'name': '2nd label origin',
     'freq': 1,
     'channels': [0],
     'mean': [100],
     'std': [10],
     'count': 0}], channels_names=['CIdU', 'IdU'])

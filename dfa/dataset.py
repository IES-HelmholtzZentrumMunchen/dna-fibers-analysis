"""
Example dataset management of the DNA fiber analysis package.

Use this module to decompress, load and use the example dataset.
"""
import zipfile
import os

import pandas as pd
from skimage import io
import numpy as np

from dfa import _utilities as _ut


class Dataset:
    """
    This class is meant to manage example dataset for DFA project.
    """
    def __init__(self, archive, storing_path='/tmp', force_decompress=False,
                 shuffle=True):
        """
        Constructor.

        The dataset archive is first decompressed in the storing path. If the
        decompressed path already exists (for instance when the archive has
        already been decompressed there, when it has already been loaded), it
        is not decompressed again, until the force_decompress flag is set to
        True.

        :param archive: Path to dataset archive.
        :type archive: str

        :param storing_path: Path used to store the decompressed dataset.
        :type storing_path: str

        :param force_decompress: Use this flag to force the decompression.
        :type force_decompress: bool

        :param shuffle: If True, the dataset is shuffled, it is not otherwise.
        :type shuffle: bool
        """
        self.archive = archive
        self.storing_path = storing_path
        self.dataset_path = os.path.join(storing_path,
                                         os.path.splitext(
                                             os.path.basename(archive))[0])

        if force_decompress or not os.path.exists(self.dataset_path):
            self._decompress()

        self.summary = pd.read_csv(
            os.path.join(self.dataset_path, 'summary.csv'),
            index_col=list(range(3))).sort_index()

        tmp = self.summary.copy()
        if shuffle:
            tmp = tmp.sample(frac=1)

        self.image_index = tmp.index.droplevel('fiber').unique()
        self.profile_index = tmp.index.unique()

        self._n_image = 0
        self._n_profile = 0

    def _decompress(self):
        """
        Utility method used to decompress the dataset at the correct path.
        """
        with zipfile.ZipFile(self.archive) as zipfiles:
            zipfiles.extractall(path=self.storing_path)

    def next_batch(self, index, n, mapping, batch_size=None):
        """
        Get the next batch of the given size as a generator.

        :param index: Name of the index used for selecting batches.
        :type index: string

        :param n: Name of the current offset of reading
        :type n: string

        :param mapping: Function that maps an index to elements to return.
        :type mapping: function

        :param batch_size: Size of the next batch. When None, the batch size is
        set to the size of the dataset (default behaviour).
        :type batch_size: strictly positive int or None

        :return: The elements of the next batch as a generator.
        :rtype: generator
        """
        if getattr(self, n) < getattr(self, index).size:
            if batch_size is None:
                batch_size = getattr(self, index).size

            begin = getattr(self, n)
            end = getattr(self, n) + batch_size
            setattr(self, n, end)

            for index in getattr(self, index)[begin:end]:
                yield mapping(index)
        else:
            return None

    def next_image_batch(self, batch_size=None):
        """
        Get the next image batch of the given size as a generator.

        :param batch_size: Size of the next batch. When None, the batch size is
        set to the size of the dataset (default behaviour).
        :type batch_size: strictly positive int or None

        :return: Tuples of the next batch as a generator. The tuples contain
        the index, the image and the manually selected fibers.
        :rtype: generator
        """
        return self.next_batch(
            batch_size=batch_size, index='image_index', n='_n_image',
            mapping=lambda index: (
                index,
                io.imread(os.path.join(self.dataset_path, 'input',
                                       '{}-{}.tif'.format(*index))),
                _ut.read_points_from_imagej_zip(
                    os.path.join(self.dataset_path, 'fibers',
                                 '{}-{}.zip'.format(*index)))))

    def next_profile_batch(self, batch_size=None):
        """
        Get the next profile batch of the given size as a generator.

        :param batch_size: Size of the next batch. When None, the batch size is
        set to the size of the dataset (default behaviour).
        :type batch_size: strictly positive int or None

        :return: Tuples of the next batch as a generator. The tuples contain
        the index, the profiles and the data view to the ground truth.
        :rtype: generator
        """
        return self.next_batch(
            batch_size=batch_size, index='profile_index', n='_n_profile',
            mapping=lambda index: (
                index,
                np.loadtxt(os.path.join(
                    self.dataset_path, 'profiles',
                    '{}-{}-Profiles #{}.csv'.format(*index)),
                    delimiter=',', skiprows=1, usecols=(0, 1, 2)),
                self.summary.ix[index]))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=str,
                        help='Path to the dataset archive.')
    parser.add_argument('--storing-path', type=str, default='/tmp',
                        help='Path where to store the decompressed archive '
                             '(default is "/tmp").')
    parser.add_argument('--force-decompress', action='store_true',
                        help='Force the decompression of the archive, even '
                             'if the storing path is not empty (default is '
                             'not).')
    args = parser.parse_args()

    dataset = Dataset(archive=args.archive,
                      storing_path=args.storing_path,
                      force_decompress=args.force_decompress)

"""
Example dataset management of the DNA fiber analysis package.

Use this module to decompress, load and use the example dataset.
"""
import zipfile
import os

import tqdm

import pandas as pd
from skimage import io
import numpy as np

from dfa import utilities as ut


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

        Parameters
        ----------
        archive : str
            Path to dataset archive.

        storing_path : str
            Path used to store the decompressed dataset.

        force_decompress : bool
            Use this flag to force the decompression.

        shuffle : bool
            If True, the dataset is shuffled, it is not otherwise.
        """
        self.archive = archive
        self.storing_path = storing_path
        self.dataset_path = os.path.join(storing_path,
                                         os.path.splitext(
                                             os.path.basename(archive))[0])
        self.images_path = os.path.join(self.dataset_path, 'images')
        self.fibers_path = os.path.join(self.dataset_path, 'fibers')
        self.profiles_path = os.path.join(self.dataset_path, 'profiles')
        self.masks_path = os.path.join(self.dataset_path, 'masks')
        self.summary_path = os.path.join(self.dataset_path, 'summary.csv')

        if force_decompress or not os.path.exists(self.dataset_path):
            self._decompress()

        if not os.path.exists(self.masks_path):  # masks are optional
            self.masks_path = None

        self.summary = pd.read_csv(
            self.summary_path,
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
            zipfiles.extractall(
                path=os.path.join(self.storing_path,
                                  os.path.splitext(
                                      os.path.basename(self.archive))[0]))

    def next_batch(self, index, n, mapping, batch_size=None):
        """
        Get the next batch of the given size as a generator.

        Parameters
        ----------
        index : str
            Name of the index used for selecting batches.

        n : str
            Name of the current offset of reading

        mapping : (pandas.MultiIndex,) -> T
            Function that maps an index to elements to return.

        batch_size : 0 < int | None
            Size of the next batch. When None, the batch size is set to the
            size of the dataset (default behaviour).

        Yields
        ------
        T
            The next batch.
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

    def get_image_path(self, index):
        """
        Get the path to the image file corresponding to given index.

        Parameters
        ----------
        index : pandas.MultiIndex
            Index of the image.

        Returns
        -------
        str
            Path of the file.
        """
        return os.path.join(self.images_path, '{}-{}.tif'.format(*index))

    def get_mask_path(self, index):
        """
        Get the path to the mask file corresponding to given index.

        Parameters
        ----------
        index : pandas.MultiIndex
            Index of the image corresponding to the mask.

        Returns
        -------
        str
            Path to the file.
        """
        if self.masks_path is not None:
            return os.path.join(self.masks_path, '{}-{}.tif'.format(*index))
        else:
            return None

    def get_fibers_file(self, index):
        """
        Get the path to the fiber file corresponding to given index.

        Parameters
        ----------
        index : pandas.MultiIndex
            Index of the fiber.

        Returns
        -------
        str
            Path of the file.
        """
        return os.path.join(self.fibers_path, '{}-{}.zip'.format(*index))

    def get_profiles_file(self, index):
        """
        Get the path to the profiles file corresponding to given index.

        Parameters
        ----------
        index : pandas.MultiIndex
            Index of the fiber.

        Returns
        -------
        str
            Path to the file.
        """
        return os.path.join(
            self.profiles_path, '{}-{}_fiber-{}.csv'.format(*index))

    def next_image_batch(self, batch_size=None, paths_only=False):
        """
        Get the next image batch of the given size as a generator.

        Parameters
        ----------
        batch_size : strictly positive int or None
            Size of the next batch. When None, the batch size is set to the
            size of the dataset (default behaviour).

        paths_only : bool
            If True, only paths to files are returned, otherwise the files
            are red.

        Yields
        ------
        (pandas.MultiIndex, numpy.ndarray, numpy.ndarray|None,
        List[numpy.ndarray])
            The next batch (index, image, mask, fibers). The mask can be None if
            no mask has been defined.
        """
        return self.next_batch(
            batch_size=batch_size, index='image_index', n='_n_image',
            mapping=lambda index: (
                index,
                self.get_image_path(index),
                self.get_mask_path(index),
                self.get_fibers_file(index))
            if paths_only else lambda index: (
                index,
                io.imread(self.get_image_path(index)),
                io.imread(self.get_mask_path(index))
                if self.get_mask_path(index) is not None else None,
                ut.read_fibers(self.get_fibers_file(index))))

    def next_profile_batch(self, batch_size=None, paths_only=False):
        """
        Get the next profile batch of the given size as a generator.

        Parameters
        ----------
        batch_size : strictly positive int or None
            Size of the next batch. When None, the batch size is set to the
            size of the dataset (default behaviour).

        paths_only : bool
            If True, only paths to files are returned, otherwise the files
            are red.

        Yields
        ------
        (pandas.MultiIndex, numpy.ndarray, pandas.DataFrame)
            The next batch (index, profiles, analysis).
        """
        return self.next_batch(
            batch_size=batch_size, index='profile_index', n='_n_profile',
            mapping=lambda index: (
                index, self.get_profiles_file(index))
            if paths_only else lambda index: (
                index,
                np.loadtxt(self.get_profiles_file(index),
                           delimiter=',', skiprows=1, usecols=(0, 1, 2)),
                self.summary.ix[index]))

    @staticmethod
    def _save(summary_path, output_path, images_path, mask_path, fibers_path,
              profiles_path, progress_bar=True):
        """
        Save dataset defined by input as zip file to given path.

        Parameters
        ----------
        summary_path : str
            Path to detailed analysis of the dataset.

        output_path : str
            Path to output file (the zip file containing dataset).

        images_path : str
            Path to the images.

        mask_path : str | None
            Path to the masks or None if there is no masks.

        fibers_path : str
            Path to the fibers.

        profiles_path : str
            Path to the profiles.

        progress_bar : bool
            If True, display a progress bar (command line); default is True.
        """
        summary = pd.read_csv(summary_path,
                              index_col=['experiment', 'image', 'fiber'])

        with zipfile.ZipFile(output_path, mode='w',
                             compression=zipfile.ZIP_DEFLATED) as archive:
            for ix in tqdm.tqdm(summary.index.droplevel('fiber').unique(),
                                desc='Compressing images, masks and fibers',
                                disable=not progress_bar):
                name = '-'.join([str(e) for e in ix])

                archive.write(
                    filename=os.path.join(images_path,
                                          '{}.tif'.format(name)),
                    arcname=os.path.join(os.path.basename(images_path),
                                         '{}.tif'.format(name)))

                if mask_path is not None:
                    archive.write(
                        filename=os.path.join(mask_path,
                                              '{}.tif'.format(name)),
                        arcname=os.path.join(os.path.basename(mask_path),
                                             '{}.tif'.format(name)))

                archive.write(
                    filename=os.path.join(fibers_path,
                                          '{}.zip'.format(name)),
                    arcname=os.path.join(os.path.basename(fibers_path),
                                         '{}.zip'.format(name)))

            for ix in tqdm.tqdm(summary.index.unique(),
                                desc='Compressing profiles',
                                disable=not progress_bar):
                name = '{}{}{}.csv'.format(
                    '-'.join([str(e) for e in ix[:-1]]),
                    ut.fiber_indicator,
                    ix[-1])

                archive.write(
                    filename=os.path.join(profiles_path, name),
                    arcname=os.path.join(os.path.basename(profiles_path), name))

            archive.write(
                filename=summary_path,
                arcname=os.path.join('summary.csv'))

    def save(self, path, progress_bar=True):
        """
        Save dataset as zip file to given path.

        Parameters
        ----------
        path : str
            Path of the zip file in which the dataset will be saved.

        progress_bar : bool
            If True, display a progress bar (command line); default is True.
        """
        Dataset._save(self.dataset_path, path, self.images_path,
                      self.masks_path, self.fibers_path, self.profiles_path,
                      progress_bar)

    @staticmethod
    def create(summary_path, images_path, fibers_path, profiles_path,
               output_path, mask_path=None, progress_bar=True):
        """
        Create a new dataset from paths to data.

        The summary is important since it is red to search the corresponding
        images, fibers and profiles in their respective paths.

        Parameters
        ----------
        summary_path : str
            Path to detailed analysis of the dataset.

        images_path : str
            Path to the images.

        fibers_path : str
            Path to the fibers.

        profiles_path : str
            Path to the profiles.

        output_path : str
            Path to output file (the zip file containing dataset).

        mask_path : str | None
            Path to the masks (or None if there is no mask, default).

        progress_bar : bool
            If True, display a progress bar (command line); default is True.
        """
        Dataset._save(summary_path, output_path, images_path, mask_path,
                      fibers_path, profiles_path, progress_bar)

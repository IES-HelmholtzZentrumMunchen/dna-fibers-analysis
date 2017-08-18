"""
Tests of the utilities module of the DNA fiber analysis package.
"""

import unittest
import os
from os import path as op
import shutil
import numpy as np
from dfa import utilities as ut


class TestUtilities(unittest.TestCase):
    def setUp(self):
        data_path = 'dfa/tests/data'

        self.directory = data_path
        self.tmp_directory = op.join(data_path, 'tmp')
        self.image_name = 'test-image'
        self.index = [2, 23]
        self.filenames = ['{}_fiber-{}.txt'.format(self.image_name, index)
                          for index in self.index]
        self.points = [
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            np.array([])]

    def tearDown(self):
        pass

    def test_write_fiber(self):
        os.mkdir(self.tmp_directory)

        try:
            for expected_index, expected_fiber, filename \
                    in zip(self.index, self.points, self.filenames):
                ut.write_fiber_from_txt(expected_fiber, self.tmp_directory,
                                        self.image_name, expected_index)
                fiber, image_name, index = ut.read_fiber_from_txt(
                    op.join(self.tmp_directory, filename))
                self.assertEqual(image_name, self.image_name)
                self.assertEqual(index, expected_index)
                np.testing.assert_allclose(fiber, expected_fiber)
        finally:
            shutil.rmtree(self.tmp_directory)

    def test_read_fiber_from_txt(self):
        for filename, expected_index, expected_fiber \
                in zip(self.filenames, self.index, self.points):
            fiber, image_name, index = ut.read_fiber_from_txt(
                op.join(self.directory, filename))
            self.assertEqual(image_name, self.image_name)
            self.assertEqual(index, expected_index)
            np.testing.assert_allclose(fiber, expected_fiber)

    def test_write_fibers(self):
        os.mkdir(self.tmp_directory)

        try:
            # test with output directory
            ut.write_fibers_from_txt(self.points, self.tmp_directory, self.image_name,
                                     indices=self.index, zipped=False)
            fibers, image_names, indices = tuple(
                zip(*ut.read_fibers_from_txt(self.tmp_directory)))

            for image_name in image_names:
                self.assertEqual(image_name, self.image_name)

            self.assertListEqual(list(indices), self.index)

            for fiber, expected_fiber in zip(fibers, self.points):
                np.testing.assert_allclose(fiber, expected_fiber)

            # test with output zip file
            ut.write_fibers_from_txt(self.points, self.tmp_directory, self.image_name,
                                     indices=self.index, zipped=True)
            fibers, image_names, indices = tuple(
                zip(*ut.read_fibers_from_txt(
                    op.join(self.tmp_directory,
                            '.'.join([self.image_name, 'zip'])))))

            for image_name in image_names:
                self.assertEqual(image_name, self.image_name)

            self.assertListEqual(list(indices), self.index)

            for fiber, expected_fiber in zip(fibers, self.points):
                np.testing.assert_allclose(fiber, expected_fiber)
        finally:
            shutil.rmtree(self.tmp_directory)

    def test_read_fibers_from_txt(self):
        # test with input directory
        red_fibers = ut.read_fibers_from_txt(self.directory)
        red_fibers = [(fiber, image_name, index)
                      for fiber, image_name, index in red_fibers
                      if image_name == self.image_name]
        self.assertEqual(len(red_fibers), 2)

        for (fiber, image_name, index), expected_fiber, expected_index \
                in zip(red_fibers, self.points, self.index):
            self.assertEqual(index, expected_index)
            np.testing.assert_allclose(fiber, expected_fiber)

        # test with input zip file
        red_fibers = ut.read_fibers_from_txt(
            op.join(self.directory, '.'.join([self.image_name, 'zip'])))
        self.assertEqual(len(red_fibers), 2)

        for (fiber, image_name, index), expected_fiber, expected_index \
                in zip(red_fibers, self.points, self.index):
            self.assertEqual(index, expected_index)
            np.testing.assert_allclose(fiber, expected_fiber)

        self.assertFalse(
            op.exists(op.join(self.directory,
                              '_'.join(['tmp', self.image_name]))))

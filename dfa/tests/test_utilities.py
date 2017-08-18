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
                ut.write_fiber(expected_fiber, self.tmp_directory,
                               self.image_name, expected_index)
                fiber, image_name, index = ut.read_fiber(
                    op.join(self.tmp_directory, filename))
                self.assertEqual(image_name, self.image_name)
                self.assertEqual(index, expected_index)
                np.testing.assert_allclose(fiber, expected_fiber)
        finally:
            shutil.rmtree(self.tmp_directory)

    def test_read_fiber(self):
        for filename, expected_index, expected_fiber \
                in zip(self.filenames, self.index, self.points):
            fiber, image_name, index = ut.read_fiber(
                op.join(self.directory, filename))
            self.assertEqual(image_name, self.image_name)
            self.assertEqual(index, expected_index)
            np.testing.assert_allclose(fiber, expected_fiber)

    def test_write_fibers(self):
        ut.write_fibers(self.points, self.directory, self.image_name,
                        indices=self.index, zipped=True)

    def test_read_fibers(self):
        red_fibers = ut.read_fibers(self.directory)
        red_fibers = [(fiber, image_name, index)
                      for fiber, image_name, index in red_fibers
                      if image_name == self.image_name]
        self.assertGreater(len(red_fibers), 0)

        for (fiber, image_name, index), expected_fiber, expected_index \
                in zip(red_fibers, self.points, self.index):
            self.assertEqual(index, expected_index)
            np.testing.assert_allclose(fiber, expected_fiber)

        red_fibers = ut.read_fibers(
            op.join(self.directory, '.'.join([self.image_name, 'zip'])))

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
                fiber, image_name, index = ut._read_fibers(
                    op.join(self.tmp_directory, filename))[0]
                self.assertEqual(image_name, self.image_name)
                self.assertEqual(index, expected_index)
                np.testing.assert_allclose(fiber, expected_fiber)
        finally:
            shutil.rmtree(self.tmp_directory)

    def test_write_fibers(self):
        def _read_and_test(path, true_image_name, true_index, true_points):
            # fibers, image_names, indices = tuple(
            #     zip(*ut.read_fibers(path)))

            fibers, image_names, indices = tuple(zip(
                *sorted(ut.read_fibers(path),
                        key=lambda element: element[-1])))

            for image_name in image_names:
                self.assertEqual(image_name, true_image_name)

            self.assertListEqual(list(indices), true_index)

            for fiber, expected_fiber in zip(fibers, true_points):
                np.testing.assert_allclose(fiber, expected_fiber)

        os.mkdir(self.tmp_directory)

        try:
            # test with output directory
            ut.write_fibers(self.points, self.tmp_directory,
                            self.image_name, indices=self.index,
                            zipped=False)
            _read_and_test(
                self.tmp_directory, self.image_name, self.index, self.points)

            # test with output zip file
            ut.write_fibers(self.points, self.tmp_directory,
                            self.image_name, indices=self.index,
                            zipped=True)
            _read_and_test(
                op.join(self.tmp_directory, '.'.join([self.image_name, 'zip'])),
                self.image_name, self.index, self.points)

            # test with output zip file and ImageJ ROI file format
            ut.write_fibers(self.points, self.tmp_directory,
                            self.image_name, indices=self.index,
                            zipped=True, roi_ij=True)
            _read_and_test(
                op.join(self.tmp_directory, '.'.join([self.image_name, 'zip'])),
                self.image_name, self.index, self.points)
        finally:
            shutil.rmtree(self.tmp_directory)

    def test__read_fibers(self):
        for filename, expected_index, expected_fiber \
                in zip(self.filenames, self.index, self.points):
            fiber, image_name, index = ut._read_fibers(
                op.join(self.directory, filename))[0]
            self.assertEqual(image_name, self.image_name)
            self.assertEqual(index, expected_index)
            np.testing.assert_allclose(fiber, expected_fiber)

    def test_read_fibers(self):
        def _test(red_fibers, size, true_points, true_index):
            self.assertEqual(len(red_fibers), size)

            fibers, image_names, indices = tuple(zip(
                *sorted(red_fibers,
                        key=lambda element: element[-1])))

            true_points, true_index = tuple(zip(
                *sorted(zip(true_points, true_index),
                        key=lambda element: element[-1])))

            for fiber, image_name, index, expected_fiber, expected_index \
                    in zip(fibers, image_names, indices,
                           true_points, true_index):
                self.assertEqual(index, expected_index)
                np.testing.assert_allclose(fiber, expected_fiber)

        # test with input directory
        red_fibers = ut.read_fibers(self.directory, image_name=self.image_name)
        _test(red_fibers, 4, self.points + self.points, self.index + self.index)

        # test with input zip file
        red_fibers = ut.read_fibers(
            op.join(self.directory, '.'.join([self.image_name, 'zip'])))
        _test(red_fibers, 2, self.points, self.index)

        self.assertFalse(
            op.exists(op.join(self.directory,
                              '_'.join(['tmp', self.image_name]))))

        # test with input zip file and ImageJ ROI file format
        red_fibers = ut.read_fibers(
            op.join(self.directory, '.'.join([self.image_name + '-ij', 'zip'])))
        _test(red_fibers, 2, self.points, self.index)

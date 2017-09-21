"""
Tests of the extraction module of the DNA fiber analysis package.
"""

import unittest
from os import path as op
from dfa import extraction as ex
from dfa import utilities as ut
from skimage import io


class TestExtraction(unittest.TestCase):
    def setUp(self):
        data_path = 'dfa/tests/data'

        self.image = io.imread(op.join(data_path, 'example.tif'))

        self.fiber = list(list(zip(*ut.read_fibers(
            data_path, 'coordinates_a0.5_b0.5_l20')))[0])

        self.fiber_image = io.imread(op.join(data_path, 'fiber_image.tif'))

        self.profiles = ex.np.loadtxt(
            op.join(data_path, 'example_profiles.csv'),
            delimiter=',', skiprows=1)

    def tearDown(self):
        pass

    def test_extract_fibers(self):
        fiber_image = ex.extract_fibers([ex.np.array([self.image, self.image])],
                                        [self.fiber])[0][0].astype('float32')

        ex.np.testing.assert_allclose(fiber_image, self.fiber_image)

    def test_extract_profiles_from_fiber(self):
        profiles = ex.extract_profiles_from_fiber(self.fiber_image,
                                                  pixel_size=0.14)

        ex.np.testing.assert_allclose(self.profiles, profiles)

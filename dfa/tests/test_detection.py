"""
Tests of the detection module of the DNA fiber analysis package.
"""

import unittest
from dfa import detection as det
from skimage import io


class TestKey(unittest.TestCase):
    def setUp(self):
        self.image = io.imread('dfa/tests/data/example.tif')

        self.mask = det.np.ones(self.image.shape).astype(bool)

        self.fiberness = {
            0.2: io.imread('dfa/tests/data/fiberness_a0.5_b0.2.tif'),
            0.5: io.imread('dfa/tests/data/fiberness_a0.5_b0.5.tif'),
            0.8: io.imread('dfa/tests/data/fiberness_a0.5_b0.8.tif')}

        self.directions = {
            0.2: det.np.array([
                io.imread('dfa/tests/data/directions_a0.5_b0.2_x.tif'),
                io.imread('dfa/tests/data/directions_a0.5_b0.2_y.tif')]),
            0.5: det.np.array([
                io.imread('dfa/tests/data/directions_a0.5_b0.5_x.tif'),
                io.imread('dfa/tests/data/directions_a0.5_b0.5_y.tif')]),
            0.8: det.np.array([
                io.imread('dfa/tests/data/directions_a0.5_b0.8_x.tif'),
                io.imread('dfa/tests/data/directions_a0.5_b0.8_y.tif')])}

        self.reconstructions = {
            10: {
                0.2: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.2_l10.tif'),
                0.5: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.5_l10.tif'),
                0.8: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.8_l10.tif')},
            20: {
                0.2: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.2_l20.tif'),
                0.5: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.5_l20.tif'),
                0.8: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.8_l20.tif')},
            40: {
                0.2: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.2_l40.tif'),
                0.5: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.5_l40.tif'),
                0.8: io.imread(
                    'dfa/tests/data/reconstruction_a0.5_b0.8_l40.tif')}}

    def tearDown(self):
        pass

    def test_fiberness_filter(self):
        for beta in self.fiberness.keys():
            fiberness, directions = det.fiberness_filter(
                self.image, [2, 3, 4], alpha=0.5, beta=beta, gamma=1)
            det.np.testing.assert_allclose(fiberness, self.fiberness[beta])
            det.np.testing.assert_allclose(directions, self.directions[beta])

    def test_reconstruct_fibers(self):
        for length in [10, 20, 40]:
            for beta in self.fiberness.keys():
                reconstruction = det.reconstruct_fibers(
                    self.fiberness[beta], self.directions[beta],
                    length=length, size=3, mask=self.mask, extent_mask=self.mask)
                det.np.testing.assert_allclose(
                    reconstruction, self.reconstructions[length][beta])

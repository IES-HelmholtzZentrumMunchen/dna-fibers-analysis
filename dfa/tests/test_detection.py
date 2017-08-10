"""
Tests of the detection module of the DNA fiber analysis package.
"""

import unittest
from os import path as op
from dfa import detection as det
from dfa import utilities as ut
from skimage import io


class TestDetection(unittest.TestCase):
    def setUp(self):
        data_path = 'dfa/tests/data'

        self.image = io.imread(op.join(data_path, 'example.tif'))

        self.mask = det.np.ones(self.image.shape).astype(bool)

        self.fiberness = {
            0.5: io.imread(op.join(data_path, 'fiberness_a0.5_b0.5.tif')),
            1.0: io.imread(op.join(data_path, 'fiberness_a0.5_b1.0.tif')),
            2.0: io.imread(op.join(data_path, 'fiberness_a0.5_b2.0.tif'))}

        self.directions = {
            0.5: det.np.array([
                io.imread(op.join(data_path, 'directions_a0.5_b0.5_x.tif')),
                io.imread(op.join(data_path, 'directions_a0.5_b0.5_y.tif'))]),
            1.0: det.np.array([
                io.imread(op.join(data_path, 'directions_a0.5_b1.0_x.tif')),
                io.imread(op.join(data_path, 'directions_a0.5_b1.0_y.tif'))]),
            2.0: det.np.array([
                io.imread(op.join(data_path, 'directions_a0.5_b2.0_x.tif')),
                io.imread(op.join(data_path, 'directions_a0.5_b2.0_y.tif'))])}

        self.reconstructions = {
            10: {
                0.5: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b0.5_l10.tif')),
                1.0: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b1.0_l10.tif')),
                2.0: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b2.0_l10.tif'))},
            20: {
                0.5: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b0.5_l20.tif')),
                1.0: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b1.0_l20.tif')),
                2.0: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b2.0_l20.tif'))},
            40: {
                0.5: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b0.5_l40.tif')),
                1.0: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b1.0_l40.tif')),
                2.0: io.imread(
                    op.join(data_path, 'reconstruction_a0.5_b2.0_l40.tif'))}}

        self.fibers = {
            10: {
                0.5: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b0.5_l10'),
                1.0: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b1.0_l10'),
                2.0: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b2.0_l10')},
            20: {
                0.5: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b0.5_l20'),
                1.0: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b1.0_l20'),
                2.0: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b2.0_l20')},
            40: {
                0.5: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b0.5_l40'),
                1.0: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b1.0_l40'),
                2.0: ut.read_points_from_txt(data_path,
                                             'coordinates_a0.5_b2.0_l40')}}

    def tearDown(self):
        pass

    def test_fiberness_filter(self):
        for beta in self.fiberness.keys():
            fiberness, directions = det.fiberness_filter(
                self.image, [2, 3, 4], alpha=0.5, beta=beta, gamma=1)
            det.np.testing.assert_allclose(fiberness, self.fiberness[beta])
            det.np.testing.assert_allclose(directions, self.directions[beta])

    def test_reconstruct_fibers(self):
        for length in self.reconstructions.keys():
            for beta in self.fiberness.keys():
                reconstruction = det.reconstruct_fibers(
                    self.fiberness[beta], self.directions[beta],
                    length=length, size=3, mask=self.mask,
                    extent_mask=self.mask)
                # noinspection PyTypeChecker
                det.np.testing.assert_allclose(
                    reconstruction, self.reconstructions[length][beta])

    # noinspection PyTypeChecker
    def test_estimate_medial_axis(self):
        for length in self.reconstructions.keys():
            for beta in self.fiberness.keys():
                coordinates = det.estimate_medial_axis(
                    self.reconstructions[length][beta])
                det.np.testing.assert_allclose(coordinates,
                                               self.fibers[length][beta])

    def test_detect_fibers(self):
        for length in self.reconstructions.keys():
            for beta in self.fiberness.keys():
                coordinates = det.detect_fibers(self.image, scales=[2, 3, 4],
                                                alpha=0.5, beta=beta,
                                                length=length, size=3,
                                                smoothing=10, min_length=30,
                                                fiberness_threshold=0.5,
                                                extent_mask=self.mask)
                det.np.testing.assert_allclose(coordinates,
                                               self.fibers[length][beta])

"""
Detection module of the DNA fiber analysis package.

Use this module to detect fibers and extract their profiles.

Note that only quite straight fibers are detected for now.
"""
import numpy as np
from skimage import feature


def enhance_fibers(image, width_range, alpha=0.5, beta=1):
    enhanced = np.zeros((len(width_range), image.shape[0], image.shape[1]))

    for index, width in enumerate(np.array(width_range) / 2):
        hxx, hxy, hyy = feature.hessian_matrix(image, sigma=width)
        l1, l2 = feature.hessian_matrix_eigvals(hxx, hxy, hyy)

        # We want to order eigenvalues by absolute values of magnitude
        swap = np.greater(np.abs(l1), np.abs(l2))
        tmp = l1[swap]
        l1[swap] = l2[swap]
        l2[swap] = tmp

        l2[l2 == 0] = np.finfo(l2.dtype).eps

        # Compute the Frangi 2D filter
        blobness = np.abs(l1) / np.abs(l2)
        structurness = np.sqrt(np.power(l1, 2) + np.power(l2, 2))

        enhanced[index, :, :] = \
            np.exp(-np.power(blobness, 2) / (2 * alpha**2)) * \
            (1 - np.exp(-np.power(structurness, 2) / (2 * beta**2)))
        enhanced[index, l2 > 0] = np.finfo(enhanced.dtype).eps

    return enhanced.max(axis=0)

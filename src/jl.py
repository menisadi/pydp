import numpy as np


def johnson_lindenstrauss_bound(samples, eps):
    return int(np.ceil(np.log(samples) / (eps**2 / 2 - eps**3 / 3)))


def johnson_lindenstrauss_transform(points, original_dimension, target_dimension):
    """
    Johnson Lindenstrauss transform
    low-distortion embeddings of points from high-dimensional into low-dimensional Euclidean space
    :param points: set of point in R^d space when d = original_dimension - numpy array
    :param original_dimension: the dimension from which the points where taken
    :param target_dimension: the target dimension
    :return: set of points in R^k space when k = target_dimension - numpy array
    """

    normal_matrix = np.random.normal(0, 1, original_dimension*target_dimension)
    normal_matrix = normal_matrix.reshape(target_dimension, original_dimension)
    return np.array([np.dot(normal_matrix, p.transpose())/np.sqrt(target_dimension) for p in points])

# quick test - delete when done
# b = johnson_lindenstrauss_bound(200, 0.5)
# print b
# xs = np.array([np.random.exponential(1, 65) for i in xrange(200)])
# nxs = johnson_lindenstrauss_transform(xs, 65, 30)


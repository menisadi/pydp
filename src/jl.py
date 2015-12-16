import numpy as np
from sklearn.random_projection import johnson_lindenstrauss_min_dim


def test_transform(samples, orig_dim, new_dim, diff, iters):
    success = 0
    for i in xrange(iters):
        xs = np.array([np.random.exponential(1, orig_dim) for i in xrange(samples)])
        nxs = johnson_lindenstrauss_transform(xs, orig_dim, new_dim)
        max_dist = points_distance_compare(xs, nxs)
        if max_dist-1 <= diff:
            success += 1
    return success/float(iters)


# wrote and than found sklearn
def johnson_lindenstrauss_bound(samples, miu):
    return int(16*np.log(2*samples)/(miu**2))


def johnson_lindenstrauss_transform_init(original_dimension, target_dimension):
    """
    Johnson Lindenstrauss transform
    low-distortion embeddings of points from high-dimensional into low-dimensional Euclidean space
    :param original_dimension: the dimension from which the points where taken
    :param target_dimension: the target dimension
    :return: instance of jl transform
    that gets set of points in R^d space when d = original_dimension as an numpy array
    and returns a projected set in R^k space when k = target_dimension as numpy array
    """
    normal_matrix = np.random.normal(0, 1, original_dimension*target_dimension)
    normal_matrix = normal_matrix.reshape(target_dimension, original_dimension)
    return lambda points: np.array([np.dot(normal_matrix, p.transpose())/np.sqrt(target_dimension) for p in points])


def johnson_lindenstrauss_transform(points, original_dimension, target_dimension):
    """
    Johnson Lindenstrauss transform
    low-distortion embeddings of points from high-dimensional into low-dimensional Euclidean space
    :param points: set of point in R^d space when d = original_dimension : numpy array
    :param original_dimension: the dimension from which the points where taken
    :param target_dimension: the target dimension
    :return: projected set of points in R^k space when k = target_dimension : numpy array
    """
    normal_matrix = np.random.normal(0, 1, original_dimension*target_dimension)
    normal_matrix = normal_matrix.reshape(target_dimension, original_dimension)
    return np.array([np.dot(normal_matrix, p.transpose())/np.sqrt(target_dimension) for p in points])


def points_distance_compare(old_data, new_data):
    data1_distances = [np.linalg.norm(x-y) for x in old_data for y in old_data]
    data2_distances = [np.linalg.norm(x-y) for x in new_data for y in new_data]
    return max((data2_distances[d]/float(data1_distances[d]))**2
               for d in xrange(len(old_data)) if data1_distances[d])

# quick test - delete when done
# s = 50
# d = 1000
# miu = 0.3
# k = johnson_lindenstrauss_min_dim(s, miu)
# if k > d:
#     raise ValueError("can't embed into smaller dimension")
# print test_transform(s, d, k, miu, 100)

import good_center as gc
import good_radius as gr
from scipy.spatial.distance import euclidean


def find(data, dimension, domain, desired_amount_of_points, approximation, failure, eps, delta,
         shrink=False, use_filter=False, return_ball=False):
    # TODO docstring
    """

    :param data:
    :param dimension:
    :param domain:
    :param desired_amount_of_points:
    :param approximation:
    :param failure:
    :param eps:
    :param delta:
    :param shrink:
    :param use_filter:
    :param return_ball:
    :return:
    """
    sample_number = len(data)
    radius = gr.find(data, domain, desired_amount_of_points, failure, eps)
    if not radius:
        print 'radius is 0'
    center = gc.find(data, sample_number, dimension, radius, desired_amount_of_points,
                     failure, approximation, eps, delta, shrink, use_filter)
    result = radius, center
    if return_ball:
        ball = [p for p in data if euclidean(p, center) <= radius]
        result = radius, center, ball

    return result


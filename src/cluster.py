import good_center as gc
import good_radius as gr
from scipy.spatial.distance import euclidean


def find(data, dimension, domain, desired_amount_of_points, approximation, failure, eps, delta,
         shrink=False, use_histograms=False, return_ball=False):
    # TODO the dimension parameter is redundant
    # TODO so is the domain, or maybe not?
    # TODO rename variables so that identical ones will ahave the same name in all procedures
    """
    Based on "Locating a Small Cluster Privately" by Kobbi Nissim, Uri Stemmer, and Salil Vadhan. PODS 2016.
    Given a data set, finds an approximately minimal cluster of points with approximately the desired amount of points
    :param data: list of points in R^dimension
    :param dimension: the dimension of the space which the points are taken from
    :param domain: tuple(absolute value of domain's end as int, minimum intervals in domain as float)
    :param desired_amount_of_points: the number of desired points in the resulting cluster
    :param approximation: 0 < float < 1. the approximation level of the result
    :param failure: 0 < float < 1. chances that the procedure will fail to return an answer
    :param eps: float > 0. privacy parameter
    :param delta: 1 > float > 0. privacy parameter
    :param shrink: boolean. default=False. if set to True will try to reduce the dimension to
    obtain a better answer (not relevant in dimension < 600)
    :param use_histograms: boolean. default=False. if set to True will use Theorem 2.5 from the paper
    instead of using the choosing-mechanism (as in the older versions of the paper)
    :param return_ball: boolean. default=False. if set to True will return, in addition to the
    radius and center, a list of the points from the data which are contained in the resulting cluster
    :return: the radius and the center of the resulting cluster. if return_ball=True returns also the points which
    are contained in the cluster
    """
    sample_number = len(data)
    radius = gr.find(data, domain, desired_amount_of_points, failure, eps)
    center = gc.find(data, sample_number, dimension, radius, desired_amount_of_points,
                     failure, approximation, eps, delta, shrink, use_histograms)
    result = radius, center
    if return_ball:
        ball = [p for p in data if euclidean(p, center) <= radius]
        result = radius, center, ball

    return result

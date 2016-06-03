import good_center as gc
import good_radius as gr
from scipy.spatial.distance import euclidean


def find(data, dimension, domain, desired_amount_of_points, approximation, failure, eps, delta, promise,
         return_ball=False):
    sample_number = len(data)
    radius = gr.find(data, domain, desired_amount_of_points, failure, eps, delta, promise)
    center = gc.find(data, sample_number, dimension, radius, desired_amount_of_points,
                     failure, approximation, eps, delta)
    result = radius, center
    if return_ball:
        ball = [p for p in data if euclidean(p, center) <= radius]
        result = radius, center, ball

    return result


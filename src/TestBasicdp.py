import unittest
import basicdp
import math
import numpy as np
import matplotlib.pyplot as plt
import random


class TestBasicdp(unittest.TestCase):
    def quality_median(self, data, range_element):
        """
        quality_median( data , range_element )
        :return: the "distance" of range_element from the median of the data
        """
        greater_than = sum(e > range_element for e in data)
        less_than = sum(e < range_element for e in data)
        return -max(0, len(data) / 2 - min(greater_than, less_than))

    def setUp(self):
        # TODO maybe move the details into different 'example' class
        # TODO check if this initialization should be in setup or not
        self.DOMAIN_SIZE = 200
        self.DATA_SIZE = 1000
        self.NUMBER_OF_ITERATIONS = 30
        self.rand_data = np.random.normal(0, 100, self.DATA_SIZE)
        self.eps = 0.500
        self.domain = range(-self.DOMAIN_SIZE, self.DOMAIN_SIZE)
        # Pr[quality_median(result) < quality_median(np.median(randD))-2/eps*(log(domain_size)+t)] < exp(-t)
        self.ERROR_PARAMETER = 10
        self.difference = (2 / self.eps * (math.log(self.DOMAIN_SIZE) + self.ERROR_PARAMETER))

    def __plot_test_results(self, result):
        """
        helper function to visualize the data and the quality of the tested mechanism result
        :param result: the index in the domain outputted by the mechanism
        :return: plot the data as a histogram annotate on it
        the median in green and the mechanism result in red
        """
        print "The true median is: %.2f" % np.median(self.rand_data)
        plt.hist(self.rand_data, bins=20, normed=True)
        plt.axvspan(self.domain[result] - 1, self.domain[result] + 1, color='red', alpha=0.5)
        plt.axvspan(np.median(self.rand_data) - 1, np.median(self.rand_data) + 1, color='green',
                    alpha=0.5)
        plt.show()

    def __test_mechanism(self, mechanism):
        """
        helper function for testing the utility of the basicdp mechanisms
        over a normally distributed data and mean quality function
        :param mechanism: a mechanism to be tested
        :return: an index of an element in the domain outputted by the mechanism.
        Taking the worst case from NUMBER_OF_ITERATIONS tries
        """
        worst_result = mechanism(self.rand_data, self.domain, self.quality_median, self.eps)
        # evaluate self.number_of_iterations times and save the worst case
        for k in range(self.NUMBER_OF_ITERATIONS):
            current_result = mechanism(self.rand_data, self.domain, self.quality_median, self.eps)
            if self.quality_median(self.rand_data, self.domain[current_result]) < \
                    self.quality_median(self.rand_data, self.domain[worst_result]):
                worst_result = current_result

        return worst_result

    def __test_mechanism_privacy(self, mechanism):
        """
        helper function for testing the privacy of the basicdp mechanisms
        over two neighbors data and mean quality function
        :param mechanism:  a mechanism to be tested
        :return:
        """
        neighbor_data = np.copy(self.rand_data)
        # remove random elemnt
        random.shuffle(neighbor_data)[:-1]
        # add random elemnt
        np.append(neighbor_data, random.uniform(min(self.rand_data), max(self.rand_data)))
        # check that the given data sets are indeed neighbors
        if (neighbor_data == self.rand_data).all():
            raise ValueError('Data sets are identical')
        elif not (neighbor_data == self.rand_data)[:-1].all():
            raise ValueError('Data sets differ in more than one element')

        # TODO how to use the results to check?
        first_result = mechanism(self.rand_data, self.domain, self.quality_median, self.eps)
        second_result = mechanism(neighbor_data, self.domain, self.quality_median, self.eps)
        return

    def test_noisy_max(self):
        """tests the noisy_max method
        over a normally distributed data and mean quality function
        :return: Pass if the noisy_max returns a relatively high value result
        """

        result = self.__test_mechanism(basicdp.noisy_max)
        # TODO change the print to something more accurate
        print 'The maximum likely difference between the ' \
              'mechanism result and the true median is: %.2f' % self.difference
        print "The Noisy-Max Mechanism returned: %.2f" % self.domain[result]
        print "Result quality: %d" % self.quality_median(self.rand_data, self.domain[result])

        # print and plot the results
        self.__plot_test_results(result)

        # pass if both mechanism returns a relatively high value result
        self.assertGreaterEqual(self.quality_median(self.rand_data, self.domain[result]),
                                self.quality_median(self.rand_data,
                                                    np.median(self.rand_data)) - self.difference)

    def test_exponential_mechanism(self):
        """tests the exponential_mechanism method
        over a normally distributed data and mean quality function
        :return: Pass if the exponential_mechanism returns a relatively high value result
        """

        result = self.__test_mechanism(basicdp.exponential_mechanism)
        print 'The maximum likely difference between the ' \
              'mechanism result and the true median is: %.2f' % self.difference
        print "The Exponential Mechanism returned: %.2f" % self.domain[result]
        print "Result quality: %d" % self.quality_median(self.rand_data, self.domain[result])

        # print and plot the results
        self.__plot_test_results(result)

        # pass if both mechanism returns a relatively high value result
        self.assertGreaterEqual(self.quality_median(self.rand_data, self.domain[result]),
                                self.quality_median(self.rand_data,
                                                    np.median(self.rand_data)) - self.difference)

    # TODO looks awful, fix!
    def test_dist(self):
        """tests the A_dist method
        over a laplace-distributed data and median quality function
        :return: Pass if the A_dist returns a value
        """
        data = np.random.laplace(0, 10, 1000)
        delta = 0.01
        answers_set = range(-50, 50)
        result = basicdp.a_dist(self.eps, delta, answers_set, data, self.quality_median)
        print "A_dist returned the index - %.2f - and value - %.2f" % (result, answers_set[result])
        print "the true median is: %.2f" % np.median(data)
        plt.hist(data, bins=30, normed=True)
        plt.axvspan(answers_set[result] - 1, answers_set[result] + 1, color='red', alpha=0.5)
        plt.axvspan(np.median(data) - 1, np.median(data) + 1, color='green', alpha=0.5)
        plt.show()

        '''
        spike = np.random.randint(self.DATA_SIZE, size=1)
        print 'true concept index = %i' % spike
        point_data = [0]*self.DATA_SIZE
        point_data[spike] = 1
        # alpha = 0.2
        # beta = 0.1
        delta = 0.01

        sampled_data_x = sorted(np.random.choice(self.DATA_SIZE, self.DATA_SIZE*10))
        sampled_data_y = [point_data[sampled_data_x[i]] for i in xrange(len(sampled_data_x))]
        sampled_data = [sampled_data_x, sampled_data_y]

        def point_quality(data, point):
            indexes = [i for i, e in enumerate(data[0]) if e == point]
            return sum([data[1][i] for i in indexes])

        dist_result = basicdp.a_dist(self.eps, delta, range(self.DATA_SIZE), sampled_data, point_quality)
        print 'A_dist result is: %i' % dist_result
        self.assertEqual(dist_result, spike)
        '''

if __name__ == '__main__':
    unittest.main()

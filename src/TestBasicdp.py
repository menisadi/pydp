import unittest
import basicdp
import math
import numpy as np
import matplotlib.pyplot as plt
import examples


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
        # TODO check if this initialization should be in setup or not
        self.DOMAIN_SIZE = 200
        self.DATA_SIZE = 1000
        self.NUMBER_OF_ITERATIONS = 30
        self.eps = 0.500

    def __plot_test_results(self, result, data, range_set):
        """
        helper function to visualize the data and the quality of the tested mechanism result
        :param result: the index in the domain outputted by the mechanism
        :return: plot the data as a histogram annotate on it
        the median in green and the mechanism result in red
        """
        print "The true median is: %.2f" % np.median(data)
        plt.hist(data, bins=20, normed=True)
        plt.axvspan(range_set[result] - 1, range_set[result] + 1, color='red', alpha=0.5)
        plt.axvspan(np.median(data) - 1, np.median(data) + 1, color='green',
                    alpha=0.5)
        plt.show()

    def __test_mechanism(self, mechanism, rand_data, range_set):
        """
        helper function for testing the utility of the basicdp mechanisms
        over a normally distributed data and mean quality function
        :param mechanism: a mechanism to be tested
        :return: an index of an element in the domain outputted by the mechanism.
        Taking the worst case from NUMBER_OF_ITERATIONS tries
        """

        worst_result = mechanism(rand_data, range_set, self.quality_median, self.eps)
        # evaluate self.number_of_iterations times and save the worst case
        for k in range(self.NUMBER_OF_ITERATIONS):
            current_result = mechanism(rand_data, range_set, examples.quality_median, self.eps)
            if self.quality_median(rand_data, current_result) < \
                    self.quality_median(rand_data, worst_result):
                worst_result = current_result

        return worst_result

    def test_noisy_max(self):
        """tests the noisy_max method
        over a normally distributed data and mean quality function
        :return: Pass if the noisy_max returns a relatively high value result
        """
        rand_data = examples.get_random_data(self.DATA_SIZE)
        range_set = range(-self.DOMAIN_SIZE, self.DOMAIN_SIZE)

        # Pr[quality_median(result) < quality_median(np.median(randD))-2/eps*(log(domain_size)+t)] < exp(-t)
        error_parameter = 10
        difference = (2 / self.eps * (math.log(self.DOMAIN_SIZE) + error_parameter))
        print "The maximum 'allowed' difference between the " \
            "mechanism result and the true median is: %.2f" % difference

        result = self.__test_mechanism(basicdp.noisy_max, rand_data, range_set)
        # TODO change the print to something more accurate

        print "The Noisy-Max Mechanism returned: %.2f" % result
        print "Result quality: %d\n" % self.quality_median(rand_data, result)

        # print and plot the results
        # self.__plot_test_results(result, rand_data, range_set)

        # pass if both mechanism returns a relatively high value result
        self.assertGreaterEqual(self.quality_median(rand_data, result),
                                self.quality_median(rand_data,
                                                    np.median(rand_data)) - difference)

    def test_exponential_mechanism(self):
        """tests the exponential_mechanism method
        over a normally distributed data and mean quality function
        :return: Pass if the exponential_mechanism returns a relatively high value result
        """
        rand_data = examples.get_random_data(self.DATA_SIZE)
        range_set = range(-self.DOMAIN_SIZE, self.DOMAIN_SIZE)

        # Pr[quality_median(result) < quality_median(np.median(randD))-2/eps*(log(domain_size)+t)] < exp(-t)
        error_parameter = 10
        difference = (2 / self.eps * (math.log(self.DOMAIN_SIZE) + error_parameter))
        print "The maximum 'allowed' difference between the " \
            "mechanism result and the true median is: %.2f" % difference

        result = self.__test_mechanism(basicdp.exponential_mechanism, rand_data, range_set)

        print "The Exponential Mechanism returned: %.2f" % result
        print "Result quality: %d\n" % self.quality_median(rand_data, result)

        # print and plot the results
        # self.__plot_test_results(result, rand_data, range_set)

        # pass if both mechanism returns a relatively high value result
        self.assertGreaterEqual(self.quality_median(rand_data, result),
                                self.quality_median(rand_data,
                                                    np.median(rand_data)) - difference)

    def test_dist(self):
        """tests the A_dist method
        over data sampled with replacements, from point_d data-set
        A_dist should return the mode - the only value that is labeled by 1
        :return: Pass if the A_dist returns a value and not 'bottom'
        """
        data = examples.get_random_data(self.DATA_SIZE, 'point')
        delta = 1e-6
        factor = np.log(1/delta)/self.eps # mode should appear at least this amount of times for stability
        print "the spike is at index: %d" % data.index(1)
        samples = examples.get_labeled_sample(data, self.DATA_SIZE*factor*2)
        print "the spike appears %d times in the samples" % examples.quality_point_mode(samples, data.index(1))
        answers_set = range(self.DATA_SIZE)
        result = basicdp.a_dist(samples, answers_set, examples.quality_point_mode, self.eps, delta)
        if type(result) == str:
            print "A_dist returned: %s" % result
        else:
            print "A_dist returned: %d" % result
        # pass the test if result is a value and not 'bottom'
        self.assertNotEqual(type(result), str)


if __name__ == '__main__':
    unittest.main()

import unittest
import basicdp
import math
import numpy as np
import matplotlib.pyplot as plt


class TestBasicdp(unittest.TestCase):

    def quality_median(self, data, range_element):
            greater_than = sum(e > range_element for e in data)
            less_than = sum(e < range_element for e in data)
            # return the "distance" of range_element from the middle of the data
            return -max(0, len(data)/2 - min(greater_than, less_than))

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
        self.difference = (2/self.eps*(math.log(self.DOMAIN_SIZE)+self.ERROR_PARAMETER))
        # TODO change the print to something more accurate
        print 'The maximum likely difference between the ' \
              'mechanism result and the true median is: %.2f' % self.difference

    def __plot_test_results(self, result):
        print "The true median is: %.2f" % np.median(self.rand_data)
        plt.hist(self.rand_data, bins=20, normed=True)
        plt.axvspan(self.domain[result]-1, self.domain[result]+1, color='red', alpha=0.5)
        plt.axvspan(np.median(self.rand_data)-1, np.median(self.rand_data)+1, color='green', alpha=0.5)
        plt.show()

    def __test_mechanism(self, mechanism):
        worst_result = mechanism(self.rand_data, self.domain, self.quality_median, self.eps)
        # evaluate self.number_of_iterations times and save the worst case
        for k in range(self.NUMBER_OF_ITERATIONS):
            current_result = mechanism(self.rand_data, self.domain, self.quality_median, self.eps)
            if self.quality_median(self.rand_data, self.domain[current_result]) < \
                    self.quality_median(self.rand_data, self.domain[worst_result]):
                worst_result = current_result

        return worst_result

    def test_noisy_max(self):
        """tests the exponential_mechanism method
        over a normally distributed data and mean quality function
        :return: Pass if the exponential_mechanism returns a relatively high value result
        """

        result = self.__test_mechanism(basicdp.noisy_max)
        print "The Noisy-Max Mechanism returned: %.2f" % self.domain[result]
        print "Result quality: %d" % self.quality_median(self.rand_data, self.domain[result])

        # for later deletion (print and plot the results)
        self.__plot_test_results(result)

        # pass if both mechanism returns a relatively high value result
        self.assertGreaterEqual(self.quality_median(self.rand_data, self.domain[result]),
                                self.quality_median(self.rand_data, np.median(self.rand_data)) - self.difference)

    def test_exponential_mechanism(self):
        """tests the exponential_mechanism method
        over a normally distributed data and mean quality function
        :return: Pass if the exponential_mechanism returns a relatively high value result
        """

        result = self.__test_mechanism(basicdp.exponential_mechanism)
        print "The Exponential Mechanism returned: %.2f" % self.domain[result]
        print "Result quality: %d" % self.quality_median(self.rand_data, self.domain[result])

        # for later deletion (print and plot the results)
        self.__plot_test_results(result)

        # pass if both mechanism returns a relatively high value result
        self.assertGreaterEqual(self.quality_median(self.rand_data, self.domain[result]),
                                self.quality_median(self.rand_data, np.median(self.rand_data)) - self.difference)

if __name__ == '__main__':
    unittest.main()

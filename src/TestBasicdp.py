import unittest
import basicdp
import math
import numpy as np
import matplotlib.pyplot as plt


class TestBasicdp(unittest.TestCase):

    def test_mechanisms(self):
        """tests the exponential_mechanism and the noisy_max methods
        over a normally distributed data and median quality function
        :return: Pass if both mechanisms returns a relatively high value result
        """
        domain_size = 200
        data_size = 1000
        eps = 0.500

        def quality_median(data, range_element):
            greater_than = sum(e > range_element for e in data)
            less_than = sum(e < range_element for e in data)
            # return the "distance" of range_element from the middle of the data
            return -max(0, len(data)/2 - min(greater_than, less_than))
            # -abs(greater_than - less_than)

        rand_data = np.random.normal(0, 100, data_size)
        domain = range(-domain_size, domain_size)
        difference = (2/eps*(math.log(domain_size)+10))
        print 'The maximum likely difference between the ' \
              'mechanism result and the true median is: %.2f' % difference

        # TODO simplify the next 3 lines
        value_of_median = sorted(rand_data)[len(rand_data)/2]
        worst_i = np.where(rand_data == value_of_median)[0][0]
        worst_j = worst_i
        # evaluate 100 times and save the worst case
        for k in range(30):
            i = basicdp.noisy_max(rand_data, domain, quality_median, eps)
            j = basicdp.exponential_mechanism(rand_data, domain, quality_median, eps)
            if quality_median(rand_data, domain[i]) < quality_median(rand_data, domain[worst_i]):
                worst_i = i
            if quality_median(rand_data, domain[j]) < quality_median(rand_data, domain[worst_j]):
                worst_j = j

        # for later deletion (print and plot)
        print "The true median is: %.2f" % np.median(rand_data)
        print "The Noisy-Max mechanism returned: %.2f" % domain[worst_i]
        print "The Exponential Mechanism returned: %.2f" % domain[worst_j]
        plt.hist(rand_data, bins=20, normed=True)
        plt.axvspan(domain[worst_i]-1, domain[worst_i]+1, color='red', alpha=0.5)
        plt.axvspan(domain[worst_j]-1, domain[worst_j]+1, color='orange', alpha=0.5)
        plt.axvspan(np.median(rand_data)-1, np.median(rand_data)+1, color='green', alpha=0.5)
        plt.show()

        # pass if both mechanism returns a relatively high value result
        self.assertGreaterEqual(quality_median(rand_data, domain[worst_i]),
                                quality_median(rand_data, np.median(rand_data))-difference)
        self.assertGreaterEqual(quality_median(rand_data, domain[worst_j]),
                                quality_median(rand_data, np.median(rand_data))-difference)

'''
    def test_exponential_mechanism(self):
        """tests the exponential_mechanism method
        over a normally distributed data and mean quality function
        :return: Pass if the exponential_mechanism returns a relatively high value result
        """
        domain_size = 200
        data_size = 1000
        eps = 0.500
        quality_median = lambda a, b: -abs(np.median(a) - b)

        rand_data = np.random.normal(0, 100, data_size)
        difference = (2/eps*(math.log(domain_size)+10))
        print 'The maximum likely difference between the ' \
              'mechanism result and the true median is: %.2f' % difference
        i = basicdp.exponential_mechanism(rand_data, range(-domain_size, domain_size), quality_median, eps)

        # for later deletion (print and plot)
        print "The Exponential Mechanism returned: %.2f" % rand_data[i]
        print "The true median is: %.2f" % np.median(rand_data)
        plt.hist(rand_data, bins=20,normed=True)
        plt.axvspan(rand_data[i]-1,rand_data[i]+1, color='red', alpha=0.5)
        plt.axvspan(np.median(rand_data)-1, np.median(rand_data)+1, color='green', alpha=0.5)
        plt.show()

        # Pr[quality_median(result) < quality_median(np.median(randD))-2/eps*(log(domain_size)+t)] < exp(-t)
        self.assertGreaterEqual(quality_median(rand_data, rand_data[i]),
                                quality_median(rand_data, np.median(rand_data))-difference)
'''

if __name__ == '__main__':
    unittest.main()

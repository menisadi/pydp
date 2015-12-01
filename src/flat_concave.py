    import basicdp
    import math


    def evaluate(data, range_max_value, quality_function, quality_promise, approximation, eps, delta,
                 intervals_bounding, max_in_interval):
        # TODO go through variables names and see if they are more or less accurate
        # TODO and maybe change some of the 'k' 'j' 'i'
        # step 2
        # print "step 2"
        log_of_range = int(math.ceil(math.log(range_max_value, 2)))
        range_max_value_tag = 2 ** log_of_range

        def extended_quality_function(data_base, j):
            if range_max_value < j <= range_max_value_tag:
                return min(0, quality_function(data_base, range_max_value))
            else:
                return quality_function(data_base, j)

        # step 4
        # print "step 4"

        def recursive_quality_function(data_base, j):
            return min(intervals_bounding(data_base, range_max_value_tag, j) - (1 - approximation) * quality_promise,
                       quality_promise-intervals_bounding(data_base, range_max_value_tag, j + 1))

        # step 6
        # print "step 6"
        recursion_returned = basicdp.exponential_mechanism(data, range(log_of_range+1), recursive_quality_function, eps)

        good_interval = 8 * (2 ** recursion_returned)
        # print "good interval: %d" % good_interval

        # step 7
        # print "step 7"
        first_intervals = [(i, i + good_interval) for i in xrange(0, range_max_value_tag, good_interval)]
        second_intervals = [(i, i + good_interval) for i in xrange(good_interval/2, range_max_value_tag, good_interval)]

        # step 9 ( using 'dist' algorithm)
        # print "step 9"
        first_chosen_interval = basicdp.a_dist(data, first_intervals, max_in_interval, eps, delta)
        second_chosen_interval = basicdp.a_dist(data, second_intervals, max_in_interval, eps, delta)

        # print type(first_chosen_interval)
        # print type(second_chosen_interval)
        if type(first_chosen_interval) == str or type(second_chosen_interval) == str:
            raise ValueError("stability problem, try taking more samples!")

        # step 10
        # print "step 10"
        first_chosen_interval_as_list = range(first_chosen_interval[0], first_chosen_interval[1]+1)
        second_chosen_interval_as_list = range(second_chosen_interval[0], second_chosen_interval[1]+1)
        return basicdp.exponential_mechanism(data, first_chosen_interval_as_list + second_chosen_interval_as_list,
                                             extended_quality_function, eps)


import numpy as np

def iterate_shirley_background(counts):
    shirley_parameter = np.mean(counts[:10])
    sum_counts = np.sum(counts)
    sum_past_point = []
    for n in range(0, len(counts)):
        sum_past_point.append(np.sum(counts[n:]))
    sum_past_point = np.array(sum_past_point)
    return counts - shirley_parameter*sum_past_point/sum_counts

def shirley_background_subtraction(counts, n_iterations):
    flat_background = np.mean(counts[-10:])
    results = [counts - flat_background]
    for n in range(0, n_iterations):
        results.append(iterate_shirley_background(results[-1]))
    return results

def shirley_background(counts, n_iterations):
    corrected_counts = shirley_background_subtraction(counts, n_iterations)
    return [counts - corrected for corrected in corrected_counts]
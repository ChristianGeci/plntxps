import numpy as np

def iterate_shirley_background(counts, step_height):
    sum_counts = np.sum(counts)
    sum_past_point = []
    for n in range(0, len(counts)):
        sum_past_point.append(np.sum(counts[n:]))
    sum_past_point = np.array(sum_past_point)
    return counts - step_height*sum_past_point/sum_counts

def shirley_background_subtraction(counts, flat_background, n_iterations):
    results = [counts - flat_background]
    for n in range(0, n_iterations):
        counts = results[-1]
        step_height = np.mean(counts[:10])
        results.append(iterate_shirley_background(counts, step_height))
    return results

def shirley_background(counts, n_iterations):
    flat_background = np.mean(counts[-10:])
    corrected_counts = shirley_background_subtraction(counts, flat_background, n_iterations)
    return [counts - corrected for corrected in corrected_counts]

def parametric_shirley_background_subtraction(counts, flat_background, step_height, n_iterations):
    results = [counts - flat_background]
    for n in range(0, n_iterations):
        counts = results[-1]
        results.append(iterate_shirley_background(counts, step_height))
    return results

def parametric_shirley_background(y, flat_background, step_height):
    corrected_counts = parametric_shirley_background_subtraction(
        y, flat_background, step_height, 5
    )
    return y - corrected_counts[-1]
import numpy as np

def sum_past_point(n, spectrum, background):
    return np.sum(spectrum.counts[n:]) - np.sum(background[n:])

def avg_spectrum_start(spectrum):
    return np.mean(spectrum.counts[:5])
def avg_spectrum_end(spectrum):
    return np.mean(spectrum.counts[-5:])

def get_flat_background(spectrum):
    return np.array([avg_spectrum_end(spectrum)] * len(spectrum.eV))

def iterate_shirley_background(spectrum, old_background):
    flat_background = get_flat_background(spectrum)
    total_sum = sum_past_point(0, spectrum, old_background)
    delta_counts = avg_spectrum_start(spectrum) - avg_spectrum_end(spectrum)
    result = []
    for n in range(0, len(spectrum.counts)):
        result.append(sum_past_point(n, spectrum, old_background))
    result = np.array(result)
    result /= total_sum
    result *= delta_counts
    result += flat_background
    return result

def shirley_background(spectrum, iteration_count):
    flat_background = get_flat_background(spectrum)
    shirley_backgrounds = [iterate_shirley_background(spectrum, flat_background)]
    for n in range(0, iteration_count - 1):
        shirley_backgrounds.append(iterate_shirley_background(
            spectrum, shirley_backgrounds[-1]))
    
    goodness_of_fit_params = [
        np.sum(np.pow(np.abs(np.array(spectrum.counts - background)), 0.1))
        for background in shirley_backgrounds]
    
    best_fit = np.argmin(goodness_of_fit_params)
    
    return shirley_backgrounds[best_fit]
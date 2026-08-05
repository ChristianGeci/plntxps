import numpy as np

def next_shirley_background(counts, step_height, flat_background):
    sum_past_point = []
    for n in range(0, len(counts)):
        sum_past_point.append(np.sum(counts[n+1:]))
    return np.array(sum_past_point) * step_height / np.sum(counts) + flat_background

def shirley_background(counts, n_iterations):
    flat_background = np.mean(counts[-10:])
    backgrounds = [flat_background]
    step_height = np.mean(counts[:5]) - flat_background
    for n in range(0, n_iterations):
        backgrounds.append(
            next_shirley_background(
                counts - backgrounds[-1],
                step_height,
                flat_background
            )
        )
    return backgrounds[-1]

def parametric_shirley_background(y, flat_background, step_height):
    backgrounds = [flat_background]
    for n in range(0, 3):
        backgrounds.append(
            next_shirley_background(
                y - backgrounds[-1],
                step_height,
                flat_background
            )
        )
    return backgrounds[-1]
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from importlib.resources import files
from io import StringIO
import copy
from collections import Counter

photoemission_path = files('plntxps.resources').joinpath('HandbookXPS_formatted.csv')
auger_path = files('plntxps.resources').joinpath('HandbookAES_formatted.csv')

photoemission_csv = photoemission_path.read_text(encoding='utf-8')
auger_csv = auger_path.read_text(encoding='utf-8')

photoemission_df = pd.read_csv(StringIO(photoemission_csv), sep = "\t")
auger_df = pd.read_csv(StringIO(auger_csv), sep = "\t")

PHOTON_ENERGY = {
    "Mg": 1253.6,
    "Al": 1486.6,
}
WORK_FUNCTION = 4.543

def kinetic_to_binding(kinetic_energy,
        photon_energy, work_function):
    return photon_energy - work_function - kinetic_energy

def get_auger_positions_and_names(element, photon_energy, work_function, df = auger_df):
    filtered_df = df[
        df['Element'].apply(lambda x: x.lower()) == element.lower()]
    positions = [float(item) for item in filtered_df["Kinetic Energy"].tolist()]
    positions  = list(kinetic_to_binding(np.array(positions),
        photon_energy, work_function))
    levels = (filtered_df['Level']
        .apply(lambda level: f"{element} {level}")).tolist()
    return positions, levels

def get_core_positions_and_names(element, df = photoemission_df):
    filtered_df = df[
        df['Element'].apply(lambda x: x.lower()) == element.lower()]
    positions = [float(item) for item in filtered_df["Binding Energy"].tolist()]
    levels = (filtered_df['Level']
        .apply(lambda level: f"{element} {level}")).tolist()
    return positions, levels

def old_plot_peaks(element, vline_min, vline_max,
        photon_energy = 1253.6, work_function = 4.454,
        minimum_distance = 100,
        **kwargs):
    core_positions, core_names = get_core_positions_and_names(element)
    auger_positions, auger_names = get_auger_positions_and_names(element,
        photon_energy, work_function)

    positions = core_positions + auger_positions
    names = core_names + auger_names

    line = plt.vlines(positions, vline_min, vline_max, **kwargs)
    adjusted_positions = simulate_node_repulsion(
        positions, minimum_distance, minimum_distance/10)
    for position, name in zip(adjusted_positions, names):
        plt.annotate(name, (position, vline_max), rotation = 45, color = line.get_color())

def max_within_range(eV, counts, position, _range):
    tuples = [
        (point_eV, point_count) for (point_eV, point_count) in tuple(zip(eV, counts))
        if np.abs(point_eV - position) <= _range
    ]
    eV_slice, count_slice = tuple(zip(*tuples))
    return np.max(count_slice)

def identify_doublets(peak_names):
    base_name_lookup = {}
    for name in peak_names:
        base_name_lookup[name] = re.sub(r"\d/\d", "", name)
    base_names = set(list(base_name_lookup.values()))
    counts = Counter(list(base_name_lookup.values()))
    doublets = []
    for name in base_names:
        if counts[name] > 1:
            doublets.append(name)
    return doublets

def plot_peaks(element, mpl_line, offset, height,
        photon_energy = PHOTON_ENERGY['Mg'], work_function = WORK_FUNCTION,
        minimum_distance = 100, hover_range = 10,
        include_names = True, shift = 0,
        doublet_coalescence_threshold = 1,
        **kwargs):
    core_positions, core_names = get_core_positions_and_names(element)
    auger_positions, auger_names = get_auger_positions_and_names(element,
        photon_energy, work_function)

    positions = core_positions + auger_positions
    names = core_names + auger_names
    print(identify_doublets(names)) # debug

    # filter out lines outside the spectrum
    positions, names = tuple(zip(*[
        (position, name) for (position, name) in zip(positions, names)
        if  position >= min(mpl_line.get_data()[0])
        and position <= max(mpl_line.get_data()[0])
    ]))
    
    # plot ticmarks above the trace of the spectrum
    vline_mins = []
    for x_position in positions:
        vline_mins.append(max_within_range(
            eV = mpl_line.get_data()[0],
            counts = mpl_line.get_data()[1],
            position = x_position,
            _range = hover_range,
        ))
    vline_mins = np.array(vline_mins) + offset
    vline_maxs = vline_mins + height

    line = plt.vlines(positions, vline_mins, vline_maxs, **kwargs)
    if not include_names:
        return
    adjusted_positions = simulate_node_repulsion(
        positions, minimum_distance, minimum_distance/10)
    for x_position, y_position, name in zip(adjusted_positions, vline_maxs, names):
        plt.annotate(name, (x_position, y_position), rotation = 45, color = line.get_color())

def simulate_node_repulsion(initial_positions, minimum_distance, granularity):
    def velocity_from_one_point(point_position, neighbor_position):
        distance_vector = neighbor_position - point_position
        if np.abs(distance_vector) >= minimum_distance:
            return 0
        if distance_vector == 0 :
            return 0
        direction = -np.sign(distance_vector)
        magnitude = np.min([minimum_distance, np.abs(1 / distance_vector * granularity)])
        return direction * magnitude
    def get_velocities(positions):
        velocities = []
        for position in positions:
            velocities.append(0)
            for neighbor in positions:
                velocities[-1] += velocity_from_one_point(position, neighbor)
        velocities = np.array(velocities)
        return velocities

    position_snapshots = [initial_positions]
    while True:
        velocities = get_velocities(position_snapshots[-1])
        if (velocities == np.zeros(len(velocities))).all():
            break
        position_snapshots.append(position_snapshots[-1] + velocities)

    return position_snapshots[-1]

def get_core_peaks_around(binding_energy, window_width = 30):
    mask = np.abs(photoemission_df["Binding Energy"] - binding_energy) <= window_width
    result = photoemission_df[mask].sort_values(by = "Binding Energy")
    return result

def get_auger_peaks_around(kinetic_energy, window_width = 30):
    mask = np.abs(auger_df["Kinetic Energy"] - kinetic_energy) <= window_width
    result = auger_df[mask].sort_values(by = "Kinetic Energy")
    return result

def get_photoemission_df():
    return photoemission_df

def get_auger_df():
    return auger_df

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from importlib.resources import files
from io import StringIO

photoemission_path = files('plntxps.resources').joinpath('HandbookXPS_formatted.csv')
auger_path = files('plntxps.resources').joinpath('HandbookAES_formatted.csv')

photoemission_csv = photoemission_path.read_text(encoding='utf-8')
auger_csv = auger_path.read_text(encoding='utf-8')

photoemission_df = pd.read_csv(StringIO(photoemission_csv), sep = "\t")
auger_df = pd.read_csv(StringIO(auger_csv), sep = "\t")

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

def plot_peaks(element, vline_min, vline_max,
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
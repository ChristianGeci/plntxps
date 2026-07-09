import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from importlib.resources import files
from io import StringIO

data_path = files('plntxps.resources').joinpath('HandbookXPS.csv')

photoemission_csv = data_path.read_text(encoding='utf-8')
photoemission_df = pd.read_csv(StringIO(photoemission_csv), sep = ";")

def get_positions_and_names(element, df):
    positions = (df[
        df['AtomicLevel.symbol'].apply(lambda x: x.lower()) == element.lower()]
        ['BindingEnergy']
        .apply(lambda x : float(re.sub(r'[^0-9.]', '', x)))
        .tolist())
    levels = (df[
        df['AtomicLevel.symbol'].apply(lambda x: x.lower()) == element.lower()]
        ['AtomicLevel.level']
        .apply(lambda x: f"{element} {x}")).tolist()
    return positions, levels

def plot_core_peaks(element, vline_min, vline_max,
        minimum_distance = 100,
        **kwargs):
    positions, names = get_positions_and_names(element, photoemission_df)
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
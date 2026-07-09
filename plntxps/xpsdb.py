import re
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

def plot_peaks(element, vline_min, vline_max, **kwargs):
    positions, names = get_positions_and_names(element, photoemission_df)
    plt.vlines(positions, vline_min, vline_max, **kwargs)
    for position, name in zip(positions, names):
        plt.annotate(name, (position, vline_max), rotation = 45)
import numpy as np
from glob import glob
import pandas as pd
import re
import lmfit
import lmfext
from lmfitxps import models
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .spectrum import Spectrum
from .background_subtraction import parametric_shirley_background
from .core import read_datafile
from os import mkdir

def boilerplate():
    plt.gca().invert_xaxis()
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Counts per Second")

def auto_shirley(params, counts):
    params['shirley_flat_background'].value = np.min(counts)
    params['shirley_step_height'].value = counts[0] - np.min(counts)

def set_satellite_param_hints(model, parent_prefix, satellite):
    model.set_param_hint(
        'amplitude', expr = f"{parent_prefix}_amplitude*{satellite['intensity']/100}")
    model.set_param_hint(
        'center', expr = f"{parent_prefix}_center-{satellite['position']}")
    model.set_param_hint('sigma', expr = f"{parent_prefix}_sigma")
    model.set_param_hint('gamma', expr = f"{parent_prefix}_gamma")
    model.set_param_hint('gaussian_sigma', expr = f"{parent_prefix}_gaussian_sigma")

def process_peak_name(peak_name):
    result = peak_name
    result = re.sub('/', '', result)
    result = re.sub(' ', '_', result)
    return result

def setup_satellite_models(peak_table: pd.DataFrame, satellites):
    fit_models = []
    for index, row in peak_table.iterrows():
        peak = process_peak_name(row['peak name'])
        if not row['has satellites']:
            continue
        for n in range(1, len(satellites)):
            satellite_model = models.ConvGaussianDoniachSinglett(
                prefix = f"{peak}_{satellites[n]['name']}_",
                independent_vars = ["x"])
            set_satellite_param_hints(satellite_model, peak, satellites[n])
            fit_models.append(satellite_model)
    return fit_models

def setup_background(bg_type):
    if bg_type == "shirley":
        return lmfit.Model(
            parametric_shirley_background,
            prefix = "shirley_",
            independent_vars = ['y'])
    elif bg_type == "tougaard":
        return models.TougaardBG(independent_vars = ["x", "y"], prefix = 'tougaard_')
    elif bg_type == "none":
        return None
    else:
        raise ValueError("Background type not recognized")

def setup_main_peaks(peak_table: pd.DataFrame):
    result = []
    for index, row in peak_table.iterrows():
        peak = process_peak_name(row['peak name'])
        result.append(models.ConvGaussianDoniachSinglett(
            prefix = peak + '_', independent_vars = ["x"]))
    return result

def setup_fit_model(peak_table: pd.DataFrame, bg_type, satellites):
    fit_models = []
    background = setup_background(bg_type)
    if background:
        fit_models.append(background)
    fit_models += setup_main_peaks(peak_table)
    if type(satellites) != type(None):
        fit_models += setup_satellite_models(peak_table, satellites)

    # merge fit models
    fit_model = fit_models[0]
    if len(fit_models) > 1:
        for model in fit_models[1:]:
            fit_model += model

    return fit_model

def setup_fit_params(peak_table: pd.DataFrame, params_path: str,
              satellites = None, bg_type = "shirley"):
    fit_model = setup_fit_model(peak_table, bg_type, satellites)
    lmfext.make_params_file(fit_model, params_path)
    return 

def plot_initial_guess(fit_model, params_path, eV, counts, guess_shirley):
    params = lmfext.read_params(params_path)
    if guess_shirley:
        auto_shirley(params, counts)
    initial_guess = fit_model.eval(
        params,
        y = counts,
        x = eV,
    )
    plt.plot(eV, counts, color = 'black', label = 'data')
    plt.plot(eV, initial_guess, ls = 'dashed', label = 'model')
    boilerplate()
    plt.legend()
    return initial_guess

def do_fit(eV, counts, fit_model, params_path, guess_shirley):
    params = lmfext.read_params(params_path)
    if guess_shirley:
        auto_shirley(params, counts)
    result = fit_model.fit(counts, params,
        x = eV, y = counts)
    return result

def group_components(components, satellites):
    component_table = {}
    # get parent peaks
    for curve_name in components.keys():
        is_satellite = False
        for satellite in satellites.values():
            if satellite['name'] in curve_name:
                is_satellite = True
        if not is_satellite:
            component_table[curve_name] = [curve_name]
    # get satellites
    for parent_curve in list(component_table.keys()):
        if "shirley" in parent_curve or "tougaard" in parent_curve:
            continue
        for n in range(1, len(satellites)):
            component_table[parent_curve].append(
                f"{parent_curve}{satellites[n]['name']}_")
    result = {}
    for parent_peak_name, child_peak_names in component_table.items():
        curves = [components[child_peak_name] for child_peak_name in child_peak_names]
        result[parent_peak_name] = np.sum(curves, axis = 0)
    return result

def plot_fit_result(eV, counts, fit_result, satellites, custom_background = None):
    components = fit_result.eval_components(x = eV, y = counts)
    plt.plot(eV, counts, color = 'black', label = 'data')
    boilerplate()
    if 'tougaard_' in components.keys():
        background = components['tougaard_']
        background_name = 'tougaard_'
    elif 'shirley_' in components.keys():
        background = components['shirley_']
        background_name = 'shirley_'
    elif type(custom_background) != type(None):
        background = custom_background
        background_name = 'custom'
    else:
        background = np.zeros(len(eV))
        background_name = 'none'
    
    if background_name == "custom":
        plt.plot(eV, fit_result.best_fit + background, label = 'fit')
        plt.plot(eV, background, label = 'background', ls = 'dashed')
    else:
        plt.plot(eV, fit_result.best_fit, label = 'fit')

    if type(satellites) != type(None):
        grouped_components = group_components(components, satellites)
    else:
        grouped_components = components
    for name, curve in grouped_components.items():
        if name != background_name and background_name != 'none':
            adjusted_curve = curve + background
        else:
            adjusted_curve = curve
        plt.plot(eV, adjusted_curve, label = name[:-1], ls = 'dashed')

    plt.legend()

def fit_procedure(
        eV: np.ndarray[float], counts: np.ndarray[float],
        peak_table: pd.DataFrame, params_path: str,
        guess_shirley: bool = False, satellites = None,
        bg_type = "shirley"):
    fit_model = setup_fit_model(peak_table, bg_type, satellites)
    result = do_fit(eV, counts, fit_model, params_path, guess_shirley)
    return result

def parabola(x, center, height, curvature):
    return curvature * (x - center)**2 + height
def _parabolic_fit(spectrum_slice: Spectrum):
    argmax = np.argmax(spectrum_slice.counts)
    center = spectrum_slice.eV[argmax]
    height = spectrum_slice.counts[argmax]

    peak_model = lmfit.Model(parabola)
    params = peak_model.make_params(
        center = center, height = height, curvature = -1)

    result = peak_model.fit(
        spectrum_slice.counts,
        params,
        x = spectrum_slice.eV,
    )
    return result
def parabolic_fit(
        spectrum: Spectrum, slice_min: float, slice_max: float,
        plot: bool):
    spectrum_slice = spectrum.slice(slice_min, slice_max)
    result = _parabolic_fit(spectrum_slice)
    if plot:
        spectrum.plot()
        plt.plot(spectrum_slice.eV, result.best_fit, ls = 'dashed')
        plt.vlines(
            [slice_min, slice_max],
            min(spectrum_slice.counts) / 1.1,
            max(spectrum_slice.counts) * 1.05,
            color = 'black', ls = 'dashed')
    return result
def fit_peak_position(
        spectrum: Spectrum, slice_min: float, slice_max: float,
        plot: bool = True) -> float:
    """
    Finds the position of a peak using a parabolic fit
    
    :param spectrum: Spectrum
    :type spectrum: Spectrum
    :param slice_min: Min of the slice that contains the peak
    :type slice_min: float
    :param slice_max: Max of the slice that contains the peak
    :type slice_max: float
    :param plot: Plot the result?
    :type plot: bool
    """
    result = parabolic_fit(spectrum, slice_min, slice_max, plot)
    return result.params['center'].value

@dataclass
class XpsBatchFit:
    experiment_table: pd.DataFrame
    region_table: pd.DataFrame
    fit_table: pd.DataFrame
    peak_tables: dict[str, pd.DataFrame]

def xps_batch_fit(experiment_table_filepath, region_table_filepath):
    experiment_table = pd.read_csv(experiment_table_filepath, sep = '\t')
    region_table = pd.read_csv(region_table_filepath, sep = '\t')

def make_blank_experiment_table(data_directory_path, experiment_table_path):
    # todo: prevent overwriting
    data_paths = glob(f"{data_directory_path}*.xy")
    with open(experiment_table_path, 'w') as f:
        f.write("filepath\tlabel\n")
        for path in data_paths:
            print(path) # debug
            f.write(f"{path}\t\n")
        f.close()
    return
def read_experiment_table(experiment_table_filepath):
    result = pd.read_csv(experiment_table_filepath, sep = '\t')
    datafiles = []
    for index, row in result.iterrows():
        datafiles.append(read_datafile(row['filepath']))
    result['data'] = datafiles
    return result

def make_blank_region_table(filepath):
    # todo: prevent overwriting
    with open(filepath, 'w') as f:
        f.write(
            'region'
            '\tparams file'
            '\tpeaks file'
            '\tslice'
            '\tdo fit'
            '\tguess shirley'
            )
        f.close()
    return
def fill_out_region_table(filepath, peaks_dir_path, params_dir_path):
    region_table = pd.read_csv(filepath, sep = '\t')
    peak_paths = []
    params_paths = []
    for index, row in region_table.iterrows():
        peak_path = f"{peaks_dir_path}/{row['region']}.csv"
        params_path = f"{params_dir_path}/{row['region']}.csv"
        peak_paths.append(peak_path)
        params_paths.append(params_path)
    region_table['params file'] = params_paths
    region_table['peaks file'] = peak_paths
    region_table.to_csv(filepath, sep = '\t', index = False)

def make_empty_fit_table(experiment_table, region_table, filepath):
    # todo: prevent overwriting
    fit_table = pd.DataFrame()
    fit_table['label'] = experiment_table['label']
    for region in region_table['region']:
        fit_table[region] = None
    fit_table['do fit'] = None
    fit_table.to_csv(filepath, sep = '\t', index = False)
    return

def make_empty_peak_tables(region_table, directory_path):
    empty_table = (
        "peak name"
        "\thas satellites"
        "\tpeak shape"
    )
    try: mkdir(directory_path)
    except FileExistsError: pass
    for region in region_table['region']:
        filepath = f"{directory_path}/{region}.csv"
        with open(filepath, 'w') as f:
            f.write(empty_table)
            f.close()
    return

def read_peak_tables(region_table):
    result = {}
    for index, row in region_table.iterrows():
        region = row['region']
        filepath = row['peaks file']
        peak_table = pd.read_csv(filepath, sep = '\t')
        result[region] = peak_table
    return result
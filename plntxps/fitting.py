import numpy as np
import pandas as pd
import re
import lmfit
import lmfext
from lmfitxps import models
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .spectrum import Spectrum
from .background_subtraction import parametric_shirley_background

def boilerplate():
    plt.gca().invert_xaxis()
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Counts per Second")

def auto_shirley(params, counts):
    params['shirley_const'].value = np.min(counts)

def read_satellite_peaks(path):
    satellites = pd.read_csv(path,
        sep = '\t').to_dict(orient='index')
    def format_satellite_name(name):
        formatted_name = re.sub(r" ", "_", name)
        formatted_name = re.sub(r",", "", formatted_name)
        return formatted_name
    for satellite in satellites.values():
        satellite['name'] = format_satellite_name(satellite['name'])
    return satellites

def set_satellite_param_hints(model, parent_prefix, satellite):
    model.set_param_hint(
        'amplitude', expr = f"{parent_prefix}_amplitude*{satellite['intensity']/100}")
    model.set_param_hint(
        'center', expr = f"{parent_prefix}_center-{satellite['position']}")
    model.set_param_hint('sigma', expr = f"{parent_prefix}_sigma")
    model.set_param_hint('gamma', expr = f"{parent_prefix}_gamma")
    model.set_param_hint('gaussian_sigma', expr = f"{parent_prefix}_gaussian_sigma")

def setup_satellite_models(peaks, satellites):
    fit_models = []
    for peak in peaks:
        for n in range(1, len(satellites)):
            satellite_model = models.ConvGaussianDoniachSinglett(
                prefix = f"{peak}_{satellites[n]['name']}_",
                independent_vars = ["x"])
            set_satellite_param_hints(satellite_model, peak, satellites[n])
            fit_models.append(satellite_model)
    return fit_models

def setup_model(peaks, bg_type, satellites):
    fit_models = []
    if bg_type == "shirley":
        fit_models.append(lmfit.Model(parametric_shirley_background, prefix = "shirley_"))
    elif bg_type == "tougaard":
        fit_models.append(models.TougaardBG(independent_vars = ["x", "y"], prefix = 'tougaard_'))
    elif bg_type == "none":
        pass
    else:
        raise ValueError("Background type not recognized")
    for peak in peaks:
        fit_models.append(models.ConvGaussianDoniachSinglett(
            prefix = peak + '_', independent_vars = ["x"]))

    if type(satellites) != type(None):
        fit_models += setup_satellite_models(peaks, satellites)

    fit_model = fit_models[0]
    if len(fit_models) > 1:
        for model in fit_models[1:]:
            fit_model += model

    return fit_model

def setup_fit(eV, counts, peaks, params_path, satellites = None, 
              plot_guess = True, bg_type = "tougaard",
              guess_shirley = False):
    fit_model = setup_model(peaks, bg_type, satellites)

    lmfext.make_params_file(fit_model, params_path)
    if plot_guess:
        plot_initial_guess(fit_model, params_path, eV, counts, guess_shirley)
    return fit_model

def plot_initial_guess(fit_model, params_path, eV, counts, guess_shirley):
    params = lmfext.read_params(params_path)
    if guess_shirley:
        auto_shirley(params, counts)
    initial_guess = fit_model.eval(
        params,
        y = counts,
        x = eV,
    )
    print("INITIAL GUESS:")
    plt.plot(eV, counts, color = 'black', label = 'data')
    plt.plot(eV, initial_guess, ls = 'dashed', label = 'model')
    boilerplate()
    plt.legend()
    plt.show()

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
        for n in range(1, len(satellites)):
            component_table[parent_curve].append(
                f"{parent_curve}{satellites[n]['name']}_")
    result = {}
    for parent_peak_name, child_peak_names in component_table.items():
        curves = [components[child_peak_name] for child_peak_name in child_peak_names]
        result[parent_peak_name] = np.sum(curves, axis = 0)
    return result

def plot_fit_result(eV, counts, fit_result, satellites, show = True, custom_background = None):
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
    if show:
        print("FIT RESULT:")
        plt.show()

def fit_procedure(eV, counts, peaks, params_path, guess_shirley = False,
        plot_guess = False, plot_result = False, satellites = None,
        bg_type = "tougaard"):
    fit_model = setup_fit(eV, counts, peaks, params_path,
        satellites = satellites, guess_shirley = guess_shirley,
        plot_guess = plot_guess, bg_type = bg_type)
    result = do_fit(eV, counts, fit_model, params_path, guess_shirley)
    if plot_result:
        plot_fit_result(eV, counts, result, satellites)
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
import numpy as np
import pandas as pd
import re
import lmfit
import lmfext
from lmfitxps import models
import matplotlib.pyplot as plt
from dataclasses import dataclass

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
        fit_models.append(models.ShirleyBG(independent_vars = ["y"], prefix = 'shirley_'))
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

def do_fit(eV, counts, fit_model, params_path, guess_shirley, plot_result = True):
    params = lmfext.read_params(params_path)
    if guess_shirley:
        auto_shirley(params, counts)
    result = fit_model.fit(counts, params,
        x = eV, y = counts)
    if plot_result:
        plot_fit_result(eV, counts, result)
    return result

def plot_fit_result(eV, counts, fit_result, show = True, custom_background = None):
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

    for name, curve in components.items():
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
    result = do_fit(eV, counts, fit_model, params_path, guess_shirley,
        plot_result = plot_result)
    return result
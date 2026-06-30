import numpy as np
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

def setup_fit(eV, counts, peaks, params_path, plot_guess = True, bg_type = "tougaard",
        guess_shirley = False):
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

    fit_model = fit_models[0]
    if len(fit_models) > 1:
        for model in fit_models[1:]:
            fit_model += model

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
        plot_guess = False, plot_result = False,
        bg_type = "tougaard"):
    fit_model = setup_fit(eV, counts, peaks, params_path, guess_shirley = guess_shirley,
        plot_guess = plot_guess, bg_type = bg_type)
    result = do_fit(eV, counts, fit_model, params_path, guess_shirley,
        plot_result = plot_result)
    return result
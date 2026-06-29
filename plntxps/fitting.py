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

def auto_shirley(params, spectrum):
    params['shirley_const'].value = np.min(spectrum.counts)

def setup_fit(spectrum, peaks, params_path, plot_guess = True, bg_type = "tougaard",
        guess_shirley = False):
    if bg_type == "shirley":
        bg = models.ShirleyBG(independent_vars = ["y"], prefix = 'shirley_')
    elif bg_type == "tougaard":
        bg = models.TougaardBG(independent_vars = ["x", "y"], prefix = 'tougaard_')
    else:
        raise ValueError("Background type not recognized")
    fit_model = bg
    for peak in peaks:
        fit_model += models.ConvGaussianDoniachSinglett(
            prefix = peak + '_', independent_vars = ["x"])

    lmfext.make_params_file(fit_model, params_path)
    if plot_guess:
        plot_initial_guess(fit_model, params_path, spectrum, guess_shirley)
    return fit_model

def plot_initial_guess(fit_model, params_path, spectrum, guess_shirley):
    params = lmfext.read_params(params_path)
    if guess_shirley:
        auto_shirley(params, spectrum)
    initial_guess = fit_model.eval(
        params,
        y = spectrum.counts,
        x = spectrum.eV,
    )
    print("INITIAL GUESS:")
    spectrum.plot(color = 'black', label = 'data')
    plt.plot(spectrum.eV, initial_guess, ls = 'dashed', label = 'model')
    boilerplate()
    plt.legend()
    plt.show()

def do_fit(spectrum, fit_model, params_path, guess_shirley, plot_result = True):
    params = lmfext.read_params(params_path)
    if guess_shirley:
        auto_shirley(params, spectrum)
    result = fit_model.fit(spectrum.counts, params,
        x = spectrum.eV, y = spectrum.counts)

    if plot_result:
        plot_fit_result(spectrum, result)

    return result

def plot_fit_result(spectrum, fit_result, show = True):
    components = fit_result.eval_components(x = spectrum.eV, y = spectrum.counts)
    spectrum.plot(color = 'black', label = 'data')
    plt.plot(spectrum.eV, fit_result.best_fit, label = 'fit')
    boilerplate()
    background = 'tougaard_'
    if 'shirley_' in components.keys():
        background = 'shirley_'
    for name, curve in components.items():
        if name != background:
            adjusted_curve = curve + components[background]
        else:
            adjusted_curve = curve
        plt.plot(spectrum.eV, adjusted_curve, label = name[:-1], ls = 'dashed')
    
    plt.legend()
    if show:
        print("FIT RESULT:")
        plt.show()


def fit_procedure(spectrum, peaks, params_path, guess_shirley = False,
        plot_guess = False, plot_result = False,
        bg_type = "tougaard"):
    fit_model = setup_fit(spectrum, peaks, params_path, guess_shirley = guess_shirley,
        plot_guess = plot_guess, bg_type = bg_type)
    result = do_fit(spectrum, fit_model, params_path, guess_shirley,
        plot_result = plot_result)
    return result
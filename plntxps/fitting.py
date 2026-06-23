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

def setup_fit(spectrum, peaks, params_path, plot_guess = True):
    bg = models.ShirleyBG(independent_vars=["y"], prefix='shirley_')
    fit_model = bg
    for peak in peaks:
        fit_model += models.ConvGaussianDoniachSinglett(
            prefix = peak + '_', independent_vars = ["x"])

    lmfext.make_params_file(fit_model, params_path)
    if plot_guess:
        plot_initial_guess(fit_model, params_path, spectrum)
    return fit_model

def plot_initial_guess(fit_model, params_path, spectrum):
    initial_guess = fit_model.eval(
    lmfext.read_params(params_path),
        y = spectrum.counts,
        x = spectrum.eV,
    )
    print("INITIAL GUESS:")
    spectrum.plot(color = 'black', label = 'data')
    plt.plot(spectrum.eV, initial_guess, ls = 'dashed', label = 'model')
    boilerplate()
    plt.legend()
    plt.show()

def do_fit(spectrum, fit_model, params_path, plot_result = True):
    params = lmfext.read_params(params_path)
    result = fit_model.fit(spectrum.counts, params,
        x = spectrum.eV, y = spectrum.counts)

    if plot_result:
        plot_fit_result(spectrum, result)

    return result

def plot_fit_result(spectrum, fit_result):
    components = fit_result.eval_components(x = spectrum.eV, y = spectrum.counts)
    spectrum.plot(color = 'black', label = 'data')
    plt.plot(spectrum.eV, fit_result.best_fit, label = 'fit')
    boilerplate()
    background = 'shirley_'
    print("FIT RESULT:")
    for name, curve in components.items():
        if name != background:
            adjusted_curve = curve + components[background]
        else:
            adjusted_curve = curve
        plt.plot(spectrum.eV, adjusted_curve, label = name[:-1], ls = 'dashed')
    
    plt.legend()
    plt.show()


def fit_procedure(spectrum, peaks, params_path,
        plot_guess = False, plot_result = False):
    fit_model = setup_fit(spectrum, peaks, params_path, 
        plot_guess=plot_guess)
    result = do_fit(spectrum, fit_model, params_path,
        plot_result = plot_result)
    return result
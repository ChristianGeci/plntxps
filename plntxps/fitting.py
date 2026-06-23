import numpy as np
import lmfit
import lmfext
from lmfitxps import models
import matplotlib.pyplot as plt

def boilerplate():
    plt.gca().invert_xaxis()
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Counts per Second")

def setup_fit(spectrum, peaks, params_path):
    bg = models.ShirleyBG(independent_vars=["y"], prefix='shirley_')
    fit_model = bg
    for peak in peaks:
        fit_model += models.ConvGaussianDoniachSinglett(
            prefix = peak + '_', independent_vars = ["x"])

    lmfext.make_params_file(fit_model, params_path)

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
    return fit_model


def do_fit(spectrum, fit_model, params_path):
    result = fit_model.fit(spectrum.counts,
        lmfext.read_params(params_path),
        x = spectrum.eV, y = spectrum.counts)
    components = result.eval_components(x = spectrum.eV, y = spectrum.counts)

    spectrum.plot(color = 'black', label = 'data')
    plt.plot(spectrum.eV, result.best_fit, label = 'fit')
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
    return result

def fit_procedure(spectrum, peaks, params_path):
    fit_model = setup_fit(spectrum, peaks, params_path)
    result = do_fit(spectrum, fit_model, params_path)
    return result
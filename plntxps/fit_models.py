import numpy as np
from lmfit.lineshapes import gaussian
from lmfit import Model
import lmfit
from lmfit.models import guess_from_peak
from scipy.signal import convolve

# much of this file is adapted from lineshapes.py in the package `lmfitxps` by Julian Andreas Hochhaus (https://github.com/Julian-Hochhaus/lmfitxps)

def fft_convolve(data, kernel):
    padding_length = min(len(data), len(kernel))
    padding = np.ones(padding_length)
    padded_data = np.concatenate((padding * data[0], data, padding * data[-1]))
    result = convolve(padded_data, kernel, mode='valid', method="fft")
    slice_start = int((len(result) - padding_length) / 2)
    truncated_result = (result[slice_start:])[:padding_length]
    return truncated_result

def normalized_gaussian_broadening(x, sigma):
    normalization_factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    gaussian_curve = gaussian(x, amplitude = 1, center = np.mean(x), sigma = sigma)
    return gaussian_curve * normalization_factor

tiny = 1.0e-15
def doniach(x, amplitude=1.0, center=0, sigma=1.0, gamma=0.0):
    arg = -(x-center)/max(tiny, sigma)
    gm1 = (1.0 - gamma)
    scale = amplitude/max(tiny, (sigma**gm1))
    return scale*np.cos(np.pi*gamma/2 + gm1*np.arctan(arg))/(1 + arg**2)**(gm1/2)

def conv_gaussian_doniach_sunjic(x, area, sigma, gamma, gaussian_sigma, center):
    conv_temp = fft_convolve(
        doniach(x, amplitude=1, center=center, sigma=sigma, gamma=gamma),
        normalized_gaussian_broadening(x, gaussian_sigma))
    return area * conv_temp / np.abs(np.trapezoid(conv_temp, x = x))

class ConvGaussianDonaichSunjic(lmfit.model.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(conv_gaussian_doniach_sunjic, *args, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('area', value=100, min=0)
        self.set_param_hint('sigma', value=0.2, min=0)
        self.set_param_hint('gamma', value=0.02, min=0.00, max = 0.3)
        self.set_param_hint('gaussian_sigma', value=0.2, min=0)
        self.set_param_hint('center', value=100, min=0)
        g_fwhm_expr = '2*{pre:s}gaussian_sigma*1.1774'
        self.set_param_hint('gaussian_fwhm', expr=g_fwhm_expr.format(pre=self.prefix))
        l_fwhm_expr = '{pre:s}sigma*(2+{pre:s}gamma*2.5135+({pre:s}gamma*3.6398)**4)'
        self.set_param_hint('lorentzian_fwhm', expr=l_fwhm_expr.format(pre=self.prefix))

    def guess(self, data, x=None, **kwargs):
        if x is None:
            return
        doniach_pars = guess_from_peak(Model(doniach), data, x, negative=False)
        gaussian_sigma = (doniach_pars["sigma"].value)
        params = self.make_params(area=doniach_pars["area"].value, sigma=doniach_pars["sigma"].value,
                                  gamma=doniach_pars["gamma"].value, gaussian_sigma=gaussian_sigma,
                                  center=doniach_pars["center"].value)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
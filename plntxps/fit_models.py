import numpy as np
from lmfit.lineshapes import doniach, gaussian, thermal_distribution
from lmfit import Model
import lmfit
from lmfit.models import guess_from_peak
from scipy.signal import convolve as sc_convolve
import scipy.constants


def fft_convolve(data, kernel, is_binding_energy=False):
    """
    Calculates the convolution of a data array with a kernel by using the convolution theorem and thereby
    transforming the time-consuming convolution operation into a multiplication of FFTs.
    The convolution using this approach is done using the `scipy.signal.convolve()` function with the `method="fft"` attribute.
    To suppress edge effects and generate a valid convolution on the full data range, the input dataset is
    extended at the edges.

    Parameters
    ----------
    data: array-like
        1D-array containing the data to convolve
    kernel: array-like
        1D-array which defines the kernel used for convolution. If binding energy scale is used, the kernel is inverted/flipped.
    is_binding_energy: boolean
        Boolean determining type of energy scale which determines the orientation of the kernel

    Returns
    ---------
    array-type
        convolution of a data array with a kernel array

    See Also
    ---------
    scipy.signal.convolve()
    """
    if is_binding_energy:
        kernel=kernel[::-1]
    min_num_pts = min(len(data), len(kernel))
    padding = np.ones(min_num_pts)
    padded_data = np.concatenate((padding * data[0], data, padding * data[-1]))
    out = sc_convolve(padded_data, kernel, mode='valid', method="fft")
    n_start_data = int((len(out) - min_num_pts) / 2)
    return (out[n_start_data:])[:min_num_pts]

def normalized_gaussian_broadening(x, sigma):
    normalization_factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    gaussian_curve = gaussian(x, amplitude = 1, center = np.mean(x), sigma = sigma)
    return gaussian_curve * normalization_factor

def singlett(x, amplitude, sigma, gamma, gaussian_sigma, center):
    is_binding_energy = x[-1] < x[0]
    conv_temp = fft_convolve(
        doniach(x, amplitude=1, center=center, sigma=sigma, gamma=gamma),
        normalized_gaussian_broadening(x, gaussian_sigma),
        is_binding_energy=is_binding_energy)
    return amplitude * conv_temp / np.abs(np.trapezoid(conv_temp, x = x))
class ConvGaussianDonaichSunjic(lmfit.model.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(singlett, *args, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('amplitude', value=100, min=0)
        self.set_param_hint('sigma', value=0.2, min=0)
        self.set_param_hint('gamma', value=0.02)
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
        params = self.make_params(amplitude=doniach_pars["amplitude"].value, sigma=doniach_pars["sigma"].value,
                                  gamma=doniach_pars["gamma"].value, gaussian_sigma=gaussian_sigma,
                                  center=doniach_pars["center"].value)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
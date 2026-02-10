import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from .charge_curve import ChargeCurve

class ChargeCurveSpline:
    """
    Wraps a charge curve and creates a callable spline function
    based on the contained charge curve
    """
    def __init__(self, charge_curve: ChargeCurve, s: list[float]):
        self.charge_curve = charge_curve
        self.t = np.arange(charge_curve.times[0], 
                             charge_curve.times[-1], 0.01)
        self.tck = sp.interpolate.splrep(
            charge_curve.times, 
            np.array(charge_curve.peak_positions), 
            s=s)
        self.spline = sp.interpolate.BSpline(*self.tck)(self.t)
    def interpolate(self, t):    
        return sp.interpolate.BSpline(*self.tck)(t)
    @property
    def start_time(self) -> float:
        return self.charge_curve.times[0]
    @property
    def end_time(self) -> float:
        return self.charge_curve.times[-1]
    def plot(self, **kwargs):
        t = np.arange(self.start_time, self.end_time, 0.01)
        binding_energies = self.interpolate(t)
        return plt.plot(t, binding_energies, **kwargs)

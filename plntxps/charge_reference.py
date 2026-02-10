from dataclasses import dataclass
import matplotlib.pyplot as plt

from .charge_curve import ChargeCurve
from .charge_curve_spline import ChargeCurveSpline

@dataclass
class ChargeReference:
    """
    Encapsulates the result of a charge referencing procedure
    """
    charge_curve_data: dict[str, ChargeCurve]
    charge_curve_splines: dict[str, ChargeCurveSpline]
    peak_positions: dict[str, float]
    charge_correction_curve: ChargeCurve

    def plot(self):
        for peak_name in self.charge_curve_data.keys():
            self.charge_curve_data[peak_name].plot()
            self.charge_curve_data[peak_name].scatter()
            self.charge_curve_splines[peak_name].plot(linestyle = 'dashed')
            plt.xlabel("Time (min)")
            plt.ylabel("Binding Energy (eV)")
            plt.title(peak_name)
            plt.show()

from dataclasses import dataclass

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
    
    def plot(self, peak_name, **kwargs):
        line, = self.charge_curve_splines[peak_name].plot(linestyle = 'dashed', **kwargs)
        self.charge_curve_data[peak_name].plot(linewidth = 0, marker = 'o', color = line.get_color())
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .peak_location import PeakLocation

@dataclass
class ChargeCurve:
    """
    Object that contains position vs time data for a particular
    XPS peak.
    """
    times: np.ndarray
    peak_positions: np.ndarray
    uncertainties: np.ndarray

    def plot(self, **kwargs):
        return plt.plot(self.times, self.peak_positions, **kwargs)
    def scatter(self, **kwargs):
        return plt.scatter(self.times, self.peak_positions, **kwargs)
    
    def slice(self, slice_start: float, slice_end: float) -> ChargeCurve:
        """
        Slice the charge curve in time and return a new charge curve object
        
        :param slice_start: Start time
        :type slice_start: float
        :param slice_end: End time
        :type slice_end: float
        :return: Sliced charge curve
        :rtype: ChargeCurve
        """
        packed_data = self.times, self.peak_positions, self.uncertainties
        transposed_data = list(zip(*packed_data))
        filtered_data = [
            datum for datum in transposed_data 
            if datum[0] > slice_start and datum[0] < slice_end]
        back_transposed_data = list(zip(*filtered_data))
        times = np.array(back_transposed_data[0])
        peak_positions = np.array(back_transposed_data[1])
        uncertainties = np.array(back_transposed_data[2])
        return ChargeCurve(times, peak_positions, uncertainties)

def charge_curve_from_tuples(
        tuples: list[tuple[float, PeakLocation]],
        start_time: float) -> ChargeCurve:
    """
    Docstring for charge_curve_from_tuples
    
    :param tuples: List of peak locations accompanied by their respective timestamps
    :type tuples: list[tuple[float, PeakLocation]]
    :param start_time: Time the experiment started
    :type start_time: float
    :return: Charge curve object
    :rtype: ChargeCurve
    """
    times = []
    peak_positions = []
    uncertainties = []
    for tuple in tuples:
        times.append(tuple[0])
        peak_positions.append(tuple[1].value)
        uncertainties.append(tuple[1].uncertainty)
    times = np.array(times) - start_time
    peak_positions = np.array(peak_positions)
    uncertainties = np.array(uncertainties)
    return ChargeCurve(times, peak_positions, uncertainties)
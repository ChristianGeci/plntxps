from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from .read_utils import (get_data, get_time, is_peak_location,
    get_region, get_comment, get_operation_name, get_center)
from .peak_location import PeakLocation

@dataclass
class Spectrum:
    "Contains an XPS spectrum"
    counts: np.ndarray
    "XPS signal"
    eV: np.ndarray
    "XPS binding energy (in eV)"
    name: str
    "Name of the XPS region (e.g. Ti 2p, O 1s, etc.)"
    comment: str
    time: float
    "Timestamp of spectrum acquisition"
    child_operations: list[Operation]
    "Operations (e.g. peak area, multi-peak fit) which reference this spectrum"
    charge_correction: float = None
    "Shift applied to binding energy to account for charging effects"
    def plot(self, ax = plt, **kwargs):
        ax.plot(self.eV, self.counts, **kwargs)

    @property
    def eV_corrected(self) -> np.ndarray:
        "Charge corrected binding energy (in eV)"
        return self.eV + self.charge_correction
    
    def slice(self, slice_min: float, slice_max: float) -> Spectrum:
        """
        Slice the spectrum object and return a new spectrum based on the slice
        
        :param slice_min: Lower binding energy range
        :type slice_min: float
        :param slice_max: Upper binding energy range
        :type slice_max: float
        :return: Sliced spectrum
        :rtype: Spectrum
        """
        packed_data = self.eV, self.counts
        transposed_data = list(zip(*packed_data))
        filtered_data = [
            datum for datum in transposed_data 
            if datum[0] > slice_min and datum[0] < slice_max]
        back_transposed_data = list(zip(*filtered_data))
        eV = np.array(back_transposed_data[0])
        counts = np.array(back_transposed_data[1])
        return Spectrum(counts=counts, eV=eV, name=None, comment=None,
                        time=None, child_operations=None, charge_correction=None)

def read_spectrum(entry: str) -> Spectrum:
    """
    Read spectrum from data file section
    
    :param entry: Section of data file that contains spectrum data and metadata
    :type entry: str
    :return: Spectrum object
    :rtype: Spectrum
    """
    eV, counts = get_data(entry)
    time = get_time(entry)
    name = get_region(entry)
    comment = get_comment(entry)
    return Spectrum(counts, eV, name, comment, time, [])

@dataclass 
class Operation:
    "Contains an operation for an XPS spectrum (e.g. peak location, peak area, etc.)"
    counts: np.ndarray
    eV: np.ndarray
    name: str
    parent: Spectrum
    parent_name: str
    peak_location: PeakLocation = None 

def read_operation(entry, parent: Spectrum) -> Operation:
    eV, counts = get_data(entry)
    name = get_operation_name(entry)
    result = Operation(counts, eV, name, parent, parent.name)
    if is_peak_location(entry):
        peak_location = get_center(entry)
        result.peak_location = peak_location
    return result
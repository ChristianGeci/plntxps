from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from .read_utils import (get_data, get_time, is_peak_location,
    get_region, get_comment, get_operation_name, get_center, get_scan_number,
    get_channel_number)
from .peak_location import PeakLocation

@dataclass
class Scan:
    """
    Contains an individual XPS scan. Note that the Scan object can correspond
    to literal scan data or separated channel data.
    """
    counts: np.ndarray
    eV: np.ndarray
    scan_number: int
    channel_number: int
    time: float

def read_scan(header, data):
    eV, counts = get_data(data)
    time = get_time(header)
    scan_number = get_scan_number(header)
    channel_number = get_channel_number(header)
    return Scan(counts, eV, scan_number, channel_number, time)

@dataclass
class Spectrum:
    "Contains an XPS spectrum"
    scans: list[Scan]
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

    def masked_counts(self, channel_mask = [], scan_mask = []):
        return np.sum([scan.counts for scan in self.scans
                       if scan.channel_number not in channel_mask
                       and scan.scan_number not in scan_mask], axis = 0)
    @property
    def counts(self):
        if hasattr(self, "_counts"):
            return self._counts
        return self.masked_counts()
    @counts.setter
    def counts(self, value):
        self._counts = value


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
        result = Spectrum(scans = [], eV=eV, name=None, comment=None,
                        time=None, child_operations=None, charge_correction=None)
        result.counts = counts
        return result

def read_spectrum(header: str, data: str) -> Spectrum:
    """
    Get spectrum object from text data

    :param header: Spectrum header block
    :type header: str   
    :param data: Spectrum data block
    :type header: str
    :return: Spectrum object
    :rtype: Spectrum
    """
    eV, counts = get_data(data) 
    time = get_time(header)
    name = get_region(header)
    comment = get_comment(header)
    scans = [read_scan(header, data)]
    return Spectrum(scans, eV, name, comment, time, [])

@dataclass 
class Operation:
    "Contains an operation for an XPS spectrum (e.g. peak location, peak area, etc.)"
    counts: np.ndarray
    eV: np.ndarray
    name: str
    parent: Spectrum
    parent_name: str
    peak_location: PeakLocation = None 

def read_operation(header: str, data: str, parent: Spectrum) -> Operation:
    """
    Get operation object from text data

    :param header: Operation header block
    :type header: str   
    :param data: Operation data block
    :type data: str   
    :param parent: Description
    :type parent: Spectrum
    :return: Description
    :rtype: Operation
    """
    # todo: verify this works
    eV, counts = get_data(data)
    name = get_operation_name(header)
    result = Operation(counts, eV, name, parent, parent.name)
    if is_peak_location(header):
        peak_location = get_center(header)
        result.peak_location = peak_location
    return result
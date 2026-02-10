from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import re
from .readutils import get_data, get_time, get_region, get_comment

@dataclass
class Spectrum:
    counts: np.ndarray
    eV: np.ndarray
    name: str
    comment: str
    time: float
    child_operations: list
    charge_correction: float = None
    def plot(self, ax = plt, **kwargs):
        ax.plot(self.eV, self.counts, **kwargs)

    @property
    def eV_corrected(self):
        return self.eV + self.charge_correction
    
    def slice(self, slice_min, slice_max):
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

def read_spectrum(entry):
    eV, counts = get_data(entry)
    time = get_time(entry)
    name = get_region(entry)
    comment = get_comment(entry)
    return Spectrum(counts, eV, name, comment, time, [])
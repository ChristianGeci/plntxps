import numpy as np
from dataclasses import dataclass

from .readutils import get_data, get_operation_name, get_center, is_peak_location
from .spectrum import Spectrum

@dataclass 
class Operation:
    counts: np.ndarray
    eV: np.ndarray
    name: str
    parent: Spectrum
    parent_name: str
    peak_location = None # Todo: fix type annotation

def read_operation(entry, _parent):
    eV, counts = get_data(entry)
    parent = _parent
    parent_name = _parent.name
    name = get_operation_name(entry)
    result = Operation(counts, eV, name, parent, parent_name)
    if is_peak_location(entry):
        peak_location = get_center(entry)
        result.peak_location = peak_location
    return result
import numpy as np
import scipy as sp
import re
from enum import Enum
from dataclasses import dataclass

def read_datafile(path: str):
    with open(path, 'r') as f:
        text = f.read()
    new_entry_pattern = r"\n\n"
    entries = re.split(new_entry_pattern, text)
    
    spectra = []
    operations = []
    for entry in entries:
        entry_type = get_entry_type(entry)
        match entry_type:
            case EntryType.SPECTRUM:
                spectra.append(read_spectrum(entry))
            case EntryType.OPERATION:
                operations.append(read_operation(entry, spectra[-1]))
                spectra[-1].child_operations.append(operations[-1])
            case _:
                pass
    return DataFile(spectra, operations)


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

def read_spectrum(entry):
    eV, counts = get_data(entry)
    time = get_time(entry)
    name = get_region(entry)
    comment = get_comment(entry)
    return Spectrum(counts, eV, name, comment, time, [])

def get_data(entry):
    pattern = r"\n\d.*(?:$|\n)"
    data_lines = re.findall(pattern, entry)
    eV = []
    counts = []
    for line in data_lines:
        parsed_line = line.split()
        eV.append(float(parsed_line[0]))
        counts.append(float(parsed_line[1]))
    eV = np.array(eV)
    counts = np.array(counts)
    return eV, counts

def get_entry_type(entry):
    spectrum_pattern = r"# Region:.*\n"
    operation_pattern = r"# Operation:.*\n"
    if re.search(spectrum_pattern, entry) is not None:
        return EntryType.SPECTRUM
    if re.search(operation_pattern, entry) is not None:
        return EntryType.OPERATION
    
class EntryType(Enum):
    SPECTRUM = 0
    OPERATION = 1

def get_time(entry): # in minutes
    pattern = r"# Acquisition Date:.*\n"
    raw_time_line = re.search(pattern, entry).group()
    time_pattern = r"\d+:\d+:\d+"
    timestamp = re.search(time_pattern, raw_time_line).group()
    parsed_timestamp = [float(n) for n in timestamp.split(":")]
    time = (parsed_timestamp[0] * 60
          + parsed_timestamp[1] 
          + parsed_timestamp[2] / 60) 
    return time

def get_region(entry):
    pattern = r"# Region:.*\n"
    region_line = re.search(pattern, entry).group()
    unwanted_part = r"# Region:\s*"
    region_name = re.sub(unwanted_part, "", region_line)
    return region_name.strip()

def get_operation_name(entry):
    pattern = r"# Operation:.*\n"
    line = re.search(pattern, entry).group()
    unwanted_part = r"# Operation:\s*"
    name = re.sub(unwanted_part, "", line)
    return name.strip()

def get_comment(entry):
    pattern = r"# Comment:.*\n"
    comment_line = re.search(pattern, entry).group()
    unwanted_part = r"# Comment:\s*"
    comment = re.sub(unwanted_part, "", comment_line)
    return comment.strip()

def get_center(entry):
    pattern = r'# Parameter: "Peak \(x\)".*\n'
    peak_center_line = re.search(pattern, entry).group()
    unwanted_part = r'# Parameter: "Peak \(x\)" = '
    center_and_uncertainty = re.sub(unwanted_part, "", peak_center_line)
    splitter = r"eV \+-"
    parsed_center_and_uncertainty = re.split(splitter, center_and_uncertainty)
    center = float(parsed_center_and_uncertainty[0])
    uncertainty = float(parsed_center_and_uncertainty[1])
    result = PeakLocation(center, uncertainty)
    return result

def is_peak_location(entry):
    pattern = r'# Operation:\s+Peak Location\n'
    if re.search(pattern, entry) is None:
        return False
    return True

@dataclass
class Spectrum:
    counts: np.ndarray
    eV: np.ndarray
    name: str
    comment: str
    time: float
    child_operations: list

@dataclass 
class Operation:
    counts: np.ndarray
    eV: np.ndarray
    name: str
    parent: Spectrum
    parent_name: str
    peak_location = None # Todo: fix type annotation

@dataclass
class PeakLocation:
    value: float
    uncertainty: float

@dataclass
class DataFile:
    spectra: list[Spectrum]
    operations: list[Operation]

    @property
    def spectrum_names(self):
        return [spectrum.name for spectrum in self.spectra]

    def list_spectra(self):
        ljust_length = 5
        for name in self.spectrum_names:
            if len(name.strip()) > ljust_length:
                ljust_length = len(name)
        header = f"{"Index":<7}{"Name":<{ljust_length+2}}{"Time (min)"}"
        print("\x1B[4m" + header + "\x1B[0m")
        for index, spectrum in enumerate(self.spectra):
            row = (f"{index:<7}"
                 + f"{spectrum.name:<{ljust_length+2}}"
                 + f"{spectrum.time:.2f}")
            print(row)
            for operation in spectrum.child_operations:
                print(f"{" " * 7}{operation.name}")
            print()
import numpy as np
import scipy as sp
import re
import matplotlib.pyplot as plt
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

def read_spectrum(entry):
    eV, counts = get_data(entry)
    time = get_time(entry)
    name = get_region(entry)
    comment = get_comment(entry)
    return Spectrum(counts, eV, name, comment, time, [])

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

class ChargeReferenceSpline:
    def __init__(self, name, times, centers, uncertainties, s):
        self.times = times
        self.centers = centers
        self.uncertainties = uncertainties
        self.t = np.arange(times[0], times[-1], 0.01)
        self.tck = sp.interpolate.splrep(times, np.array(centers), s=s)
        self.spline = sp.interpolate.BSpline(*self.tck)(self.t)
        self.name = name
    def interpolate(self, t):    
        return sp.interpolate.BSpline(*self.tck)(t)

@dataclass
class PeakLocation:
    value: float
    uncertainty: float

@dataclass
class ChargeCurve:
    times: np.ndarray
    peak_positions: np.ndarray
    uncertainties: np.ndarray

    def plot(self, **kwargs):
        plt.plot(self.times, self.peak_positions, **kwargs)
    def scatter(self, **kwargs):
        plt.scatter(self.times, self.peak_positions, **kwargs)
    
    def slice(self, slice_start, slice_end):
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
        

def charge_curve_from_tuples(tuples, start_time):
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


@dataclass
class DataFile:
    spectra: list[Spectrum]
    operations: list[Operation]
    charge_reference = None

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

    def get_charge_curve(self, region_name) -> ChargeCurve:
        times_locations = [(operation.parent.time, operation.peak_location)
            for operation in self.operations 
            if operation.peak_location != None
                and operation.parent.name == region_name]
        result = charge_curve_from_tuples(times_locations, self.spectra[0].time)
        return result 
    
    def get_charge_referenced_peak_positions(self, 
            time_slice, s = [0.02, 0.02, 0.02], Au_name = 'Au 4f',
            peaks = ['O 1s', 'Ti 2p']):
        if len(s) != len(peaks) + 1:
            raise IndexError(
                's value needed for each peak type (including gold)')
        slice_start = time_slice[0]
        slice_end = time_slice[1]
        splines = []

        Au_charge_curve = self.get_charge_curve(Au_name)
        sliced_Au_charge_curve = Au_charge_curve.slice(
            slice_start, slice_end)

        splines.append(ChargeReferenceSpline(Au_name, 
            sliced_Au_charge_curve.times,
            sliced_Au_charge_curve.peak_positions,
            sliced_Au_charge_curve.uncertainties,
            s[0]))
        
        # make all other splines
        for index, name in enumerate(peaks):
            charge_curve = self.get_charge_curve(name)
            sliced_charge_curve = charge_curve.slice(
                slice_start, slice_end)
            splines.append(ChargeReferenceSpline(name, 
                sliced_charge_curve.times,
                sliced_charge_curve.peak_positions,
                sliced_charge_curve.uncertainties, 
                s[index+1]))
        
        # find common time domain for all splines
        t_common_min = max([spline.times[0] for spline in splines])
        t_common_max = min([spline.times[-1] for spline in splines])
        t_common = np.arange(t_common_min, t_common_max, 0.01)
        
        #reference splines to the Au 4f 7/2
        charge_correction_curve = 84 - splines[0].interpolate(t_common)
        
        charge_corrected_splines = []
        for spline in splines[1:]:
            charge_corrected_splines.append(
                spline.interpolate(t_common) + charge_correction_curve)
        
        #print/store results
        charge_corrected_peak_positions = []
        for index, spline in enumerate(splines[1:]):
            print(f'{spline.name} peak position: {np.mean(charge_corrected_splines[index]):.4f} +- {np.std(charge_corrected_splines[index]):.4f}')
            charge_corrected_peak_positions.append(np.mean(charge_corrected_splines[index]))
        
        self.charge_reference = ChargeReference(
            dict(zip(splines, [Au_name] + peaks)),
            dict(zip(peaks, charge_corrected_peak_positions)))

        return

@dataclass
class ChargeReference:
    """
    Encapsulates the result of a charge referencing procedure
    """
    splines: dict[str, ChargeReferenceSpline]
    peak_postions: dict[str, float]
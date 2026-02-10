import numpy as np
import scipy as sp
import re
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass

from contextlib import contextmanager
@contextmanager
def autoscale_turned_off(ax=None):
  ax = ax or plt.gca()
  lims = [ax.get_xlim(), ax.get_ylim()]
  yield
  ax.set_xlim(*lims[0])
  ax.set_ylim(*lims[1])


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
    charge_correction: float = None
    def plot(self, ax = plt, **kwargs):
        ax.plot(self.eV, self.counts, **kwargs)

    @property
    def eV_corrected(self):
        return self.eV + self.charge_correction

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
    def __init__(self, charge_curve, s):
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
    def start_time(self):
        return self.charge_curve.times[0]
    @property
    def end_time(self):
        return self.charge_curve.times[-1]
    def plot(self, **kwargs):
        t = np.arange(self.start_time, self.end_time, 0.01)
        binding_energies = self.interpolate(t)
        return plt.plot(t, binding_energies, **kwargs)

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
        return plt.plot(self.times, self.peak_positions, **kwargs)
    def scatter(self, **kwargs):
        return plt.scatter(self.times, self.peak_positions, **kwargs)
    
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
    
    @property
    def start_time(self):
        return self.spectra[0].time

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
        result = charge_curve_from_tuples(
            times_locations, self.spectra[0].time)
        return result 
    
    def get_charge_referenced_peak_positions(self, 
            time_slice, s = [0.02, 0.02, 0.02], 
            reference_name = 'Au 4f', reference_energy = 84,
            peak_names = ['O 1s', 'Ti 2p'],
            plot_result = False):
        if len(s) != len(peak_names) + 1:
            raise IndexError(
                's value needed for each peak type (including gold)')
        slice_start = time_slice[0]
        slice_end = time_slice[1]
        splines = {}
        charge_curve_slices = {}

        # fit spline to charge curve of the reference
        reference_charge_curve = self.get_charge_curve(reference_name)
        charge_curve_slices[reference_name] = reference_charge_curve.slice(
            slice_start, slice_end)
        splines[reference_name] = (ChargeReferenceSpline( 
            charge_curve_slices[reference_name],
            s[0]))
        
        # make all other splines
        for index, name in enumerate(peak_names):
            charge_curve = self.get_charge_curve(name)
            charge_curve_slices[name] = charge_curve.slice(
                slice_start, slice_end)
            splines[name] = (ChargeReferenceSpline(
                charge_curve_slices[name],
                s[index+1]))
        
        # find common time domain for all splines
        t_common_min = max([spline.start_time for spline in splines.values()])
        t_common_max = min([spline.end_time for spline in splines.values()])
        t_common = np.arange(t_common_min, t_common_max, 0.01)
        
        # get charge shift of the reference as a function of time
        charge_correction_curve = ChargeCurve(
            times = t_common,
            peak_positions = reference_energy 
                           - splines[reference_name].interpolate(t_common),
            uncertainties = None)

        # find the charge corrected peak locations
        charge_corrected_splines = {}
        charge_corrected_peak_positions = {}
        for name in peak_names:
            # apply charge correction shift
            charge_corrected_splines[name] = (
                splines[name].interpolate(t_common) 
                + charge_correction_curve.peak_positions)
            # take mean value of corrected spline
            charge_corrected_peak_positions[name] = (
                np.mean(charge_corrected_splines[name]))
        
        # print results
        for name in peak_names:
            print(f'{name} peak position: '
                + f'{np.mean(charge_corrected_splines[name]):.4f} +- '
                + f'{np.std(charge_corrected_splines[name]):.4f}')
        
        self.charge_reference = ChargeReference(
            charge_curve_slices,
            splines,
            charge_corrected_peak_positions,
            charge_correction_curve)
        if plot_result:
            self.charge_reference.plot()
        return
    
    def charge_reference_valence_band_spectra(self, 
            time_slice, reference_spectrum, s = [0.02, 0.02], 
            reference_peaks = ['O 1s', 'Ti 2p'], valence_band_name = "Valence Band", 
            save_figures = "", plot_result = True):
        
        if len(s) != len(reference_peaks):
            raise IndexError('s value needed for each peak type')
        slice_start = time_slice[0]
        slice_end = time_slice[1]
        
        reference_peak_positions = reference_spectrum.charge_reference.peak_positions
        
        def get_reference_splines():
            reference_splines = {}
            # make splines for reference peaks
            for index, name in enumerate(reference_peaks):
                # get times, centers
                charge_curve = self.get_charge_curve(name)
                sliced_charge_curve = charge_curve.slice(slice_start, slice_end)
                # make spline
                reference_splines[name] = (ChargeReferenceSpline(
                    sliced_charge_curve, s[index]))
            return reference_splines
        reference_splines = get_reference_splines()
        
        # dictionary of charge correction splines at time t
        def charge_correction_splines(t):
            result = {}
            for name, spline in reference_splines.items():
                result[name] = (reference_peak_positions[name] - spline.interpolate(t))
            return result
        # average of all charge correction splines at time t
        def mean_charge_correction_spline(t):
            return np.mean(np.array(list(charge_correction_splines(t).values())))

        # collect all valence band spectra in time slice
        def filter_valence_band_spectra():
            included_valence_band_spectra = [x for x in self.spectra 
                if x.name == valence_band_name 
                and (slice_start <= x.time - self.start_time <= slice_end)]
            total_valence_band_spectrum_count = len([x for x in self.spectra
                if x.name == valence_band_name])
            excluded_valence_band_spectra_count = (total_valence_band_spectrum_count
                                                - len(included_valence_band_spectra))
            print(f'time slice excludes {excluded_valence_band_spectra_count} '
                + f'of {total_valence_band_spectrum_count} valence band spectra')
            return included_valence_band_spectra
        included_valence_band_spectra = filter_valence_band_spectra()
        
        # get the corresponding charge correction shift to each valence band spectrum
        for spectrum in included_valence_band_spectra:
            spectrum.charge_correction = (
                mean_charge_correction_spline(spectrum.time-self.start_time))

        def get_eV_window():
            window_cutoff = [0, 0]
            window_cutoff[0] = max([spectrum.charge_correction for spectrum in 
                                    included_valence_band_spectra])
            window_cutoff[1] = min([spectrum.charge_correction for spectrum in 
                                    included_valence_band_spectra])
            return np.arange(
                included_valence_band_spectra[0].eV[-1] + window_cutoff[0], 
                included_valence_band_spectra[0].eV[0]  + window_cutoff[1], 
                0.01)[::-1]
        eV_window = get_eV_window()
        
        # create a list of linear interpolation functions which correspond to the shifted valence band spectra
        def get_shifted_valence_band_interpolations():
            shifted_valence_band_interpolations = []
            for spectrum in included_valence_band_spectra:
                shifted_valence_band_interpolations.append(
                    sp.interpolate.interp1d(spectrum.eV_corrected, spectrum.counts))
            return shifted_valence_band_interpolations
        shifted_valence_band_interpolations = get_shifted_valence_band_interpolations()
        
        # add them all up
        shifted_valence_band_sum = []
        for binding_energy in eV_window:
            interpolated_counts = 0
            for interpolation in shifted_valence_band_interpolations:
                try:
                    interpolated_counts += interpolation(binding_energy)
                except ValueError:
                    print(f'hit value error at binding energy {binding_energy}')
            shifted_valence_band_sum.append(interpolated_counts)
        shifted_valence_band_sum = np.array(shifted_valence_band_sum)
        
        # store the data
        self.charge_corrected_valence_band = Spectrum(
            eV = eV_window, counts = shifted_valence_band_sum, 
            time = None, child_operations = None, comment = None,
            name = "charge corrected valence band")
        
        # plotting stuff
        if (not plot_result):
            return
        # plot the individual charge curves
        for peak_name in reference_peaks:
            reference_splines[peak_name].charge_curve.plot()
            reference_splines[peak_name].charge_curve.scatter()
            reference_splines[peak_name].plot(linestyle = 'dashed')
            # Todo: add error bars
            plt.title(peak_name)
            plt.show()
        # plot the charge correction curves and their mean
        def plot_charge_correction_splines():
            t_common_min = max([spline.start_time for spline in reference_splines.values()])
            t_common_max = min([spline.end_time for spline in reference_splines.values()])
            t_common = np.arange(t_common_min, t_common_max, 0.01)
            splines = charge_correction_splines(t_common)
            mean_accumulator = np.zeros(len(list(splines.values())[0]))
            for peak_name in reference_peaks:
                plt.plot(t_common, splines[peak_name], label = peak_name)
                mean_accumulator += splines[peak_name]
            mean = mean_accumulator / len(splines.values())
            plt.plot(t_common, mean, label = "Mean")
            plt.plot()
            plt.xlabel("Time (min)")
            plt.ylabel("Δ Binding Energy (eV)")
            plt.title("Charge correction curves")
            plt.legend()
            plt.show()
        plot_charge_correction_splines()

        
        #plot the sum of all valence band spectra, with and without shifting
        valence_band_spectrum = np.array(included_valence_band_spectra[0].counts)
        for spectrum in included_valence_band_spectra[1:]:
            valence_band_spectrum += np.array(spectrum.counts)
        
        fig3, ax3 = plt.subplots(1, 2)
        fig3.set_size_inches(16, 5)
        
        ax3[0].plot(eV_window, shifted_valence_band_sum, label = 'charge-corrected')
        ax3[0].plot(included_valence_band_spectra[0].eV, valence_band_spectrum, label = 'no shift applied', linestyle = 'dashed')
        ax3[0].legend()
        ax3[0].invert_xaxis()
        ax3[0].set_xlabel("binding energy (eV)")
        ax3[0].set_ylabel("counts")
        
        ax3[1].plot(eV_window, shifted_valence_band_sum, label = 'charge-corrected')
        ax3[1].plot(included_valence_band_spectra[0].eV, valence_band_spectrum, label = 'no shift applied', linestyle = 'dashed')
        ax3[1].legend()
        ax3[1].set_xlim(2, 6)
        ax3[1].invert_xaxis()
        ax3[1].set_xlabel("binding energy (eV)")
        
        return
    def find_fermi_edge_linfit(
            self, background_range, edge_range, instrumental_broadening = 0.8):

        background_min = np.amax(np.where(
            self.charge_corrected_valence_band.eV > background_range[1]))
        background_max = np.amax(np.where(
            self.charge_corrected_valence_band.eV > background_range[0]))

        edge_min = np.amax(np.where(
            self.charge_corrected_valence_band.eV > edge_range[1]))
        edge_max = np.amax(np.where(
            self.charge_corrected_valence_band.eV > edge_range[0]))

        y_background = self.charge_corrected_valence_band.counts[
            background_min:background_max]
        x_background = self.charge_corrected_valence_band.eV[
            background_min:background_max]
        X_background = np.array([[1]*len(y_background), x_background]).T
        fit_background = sp.optimize.lsq_linear(X_background, y_background)

        y_edge = self.charge_corrected_valence_band.counts[edge_min:edge_max]
        x_edge = self.charge_corrected_valence_band.eV[edge_min:edge_max]
        X_edge = np.array([[1]*len(y_edge), x_edge]).T
        fit_edge = sp.optimize.lsq_linear(X_edge, y_edge)

        fig, ax = plt.subplots()

       # ax.plot(self.charge_corrected_valence_band_eV,
       #         self.charge_corrected_valence_band_counts, color = 'tab:blue')
        self.charge_corrected_valence_band.plot(color = "tab:blue")

        with autoscale_turned_off(ax):
            ax.plot(self.charge_corrected_valence_band.eV,
                    fit_background.x[0]
                  + self.charge_corrected_valence_band.eV * fit_background.x[1],
                     color = 'tab:green', linestyle = 'dashed')
            ax.plot(self.charge_corrected_valence_band.eV,
                    fit_edge.x[0]
                  + self.charge_corrected_valence_band.eV * fit_edge.x[1],
                    color = 'tab:green', linestyle = 'dashed')
        
        
        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Counts')
        VBM_with_broadening = ((fit_edge.x[0]-fit_background.x[0])
                              /(fit_background.x[1]-fit_edge.x[1]))
        VBM = VBM_with_broadening + instrumental_broadening / 2
        
        
        ax.scatter(
            self.charge_corrected_valence_band.eV[background_min],
            fit_background.x[0]
          + self.charge_corrected_valence_band.eV[background_min] 
          * fit_background.x[1],
            color = 'tab:green', zorder = 3)
        ax.scatter(
            self.charge_corrected_valence_band.eV[background_max],
            fit_background.x[0]
          + self.charge_corrected_valence_band.eV[background_max]
          * fit_background.x[1],
            color = 'tab:green', zorder = 3)
        
        ax.scatter(
            self.charge_corrected_valence_band.eV[edge_min],
            fit_edge.x[0]
          + self.charge_corrected_valence_band.eV[edge_min]
          * fit_edge.x[1],
            color = 'tab:green', zorder = 3)
        ax.scatter(
            self.charge_corrected_valence_band.eV[edge_max],
            fit_edge.x[0]
          + self.charge_corrected_valence_band.eV[edge_max]
          * fit_edge.x[1],
            color = 'tab:green', zorder = 3)
        
        ax.scatter(
            VBM_with_broadening,
            fit_background.x[0] + VBM_with_broadening * fit_background.x[1],
            color = 'tab:red', zorder = 3)
        
        ax.vlines(
            VBM,
            fit_edge.x[0]+self.charge_corrected_valence_band.eV[edge_min]*fit_edge.x[1],
            fit_background.x[0]+self.charge_corrected_valence_band.eV[background_max]*fit_background.x[1],
            color = 'tab:red', linestyle = 'dashed')
        
        print(f'Valence Band Maximum: {VBM} eV') 
    
    def find_fermi_edge(self, smoothing_factor, guess):
        
        #spline fit:
        x = self.charge_corrected_valence_band.eV[::-1]
        y = self.charge_corrected_valence_band.counts[::-1]

        tck = sp.interpolate.splrep(x, y, s=smoothing_factor)
        y_spline = sp.interpolate.BSpline(*tck)(x)
        
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(15, 5.5)


        axs[0].plot(x, y, color = 'tab:blue')
        axs[0].plot(x, y_spline, color = 'tab:orange') 
        axs[0].vlines(guess, y.min(), y.max(), color = 'red', linestyle = 'dashed')
        axs[0].invert_xaxis()

        axs[1].plot(x, y, color = 'tab:blue')
        axs[1].plot(x, y_spline, color = 'tab:orange')
        axs[1].vlines(guess, y.min(), y.max(), color = 'red', linestyle = 'dashed')
        axs[1].set_xlim(guess-2, guess+2)
        axs[1].invert_xaxis()
        fig.suptitle('spline fit to charge-corrected data')
        
        #first derivative:
        
        fig2, axs2 = plt.subplots(1, 2)
        
        yp = np.diff(y_spline)/np.diff(x)
        xp = (x[:-1] + x[1:]) / 2 
        fig2.set_size_inches(15, 5.5)

        axs2[0].plot(xp, yp, color = 'tab:blue')
        axs2[0].vlines(guess, yp.min(), yp.max(), color = 'red', linestyle = 'dashed')
        axs2[0].invert_xaxis()
        axs2[1].plot(xp, yp, color = 'tab:blue')
        axs2[1].vlines(guess, yp.min(), yp.max(), color = 'red', linestyle = 'dashed')
        axs2[1].set_xlim(guess-2, guess+2)
        axs2[1].invert_xaxis()



        fig2.suptitle('first derivative')



        #second derivative:

        ypp = np.diff(yp)/np.diff(xp)
        xpp = (xp[:-1] + xp[1:]) / 2 

        fig3, axs3 = plt.subplots(1, 2)
        fig3.set_size_inches(15, 5.5)

        axs3[0].plot(xpp, ypp, color = 'tab:blue')
        axs3[0].hlines(0, xpp.min(), xpp.max(), color = 'red', linestyle = 'dashed')
        axs3[0].vlines(guess, ypp.min(), ypp.max(), color = 'red', linestyle = 'dashed')
        axs3[0].invert_xaxis()
        axs3[1].plot(xpp, ypp, color = 'tab:blue')
        axs3[1].hlines(0, xpp.min(), xpp.max(), color = 'red', linestyle = 'dashed')
        axs3[1].vlines(guess, ypp.min(), ypp.max(), color = 'red', linestyle = 'dashed')
        axs3[1].set_xlim(guess-2, guess+2)
        axs3[1].set_ylim(ypp.min()/2, ypp.max()/2)
        axs3[1].invert_xaxis()

        fig3.suptitle('second derivative')

@dataclass
class ChargeReference:
    """
    Encapsulates the result of a charge referencing procedure
    """
    charge_curve_data: dict[str, ChargeCurve]
    charge_curve_splines: dict[str, ChargeReferenceSpline]
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

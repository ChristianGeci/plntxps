print("data processed with plnt_xps version 0.1.1")
print("our crew is replaceable, your package isn't!")

# Processing Laborsaving iNterpolative Tools for XPS

import numpy as np
import scipy as sp
import re
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .spectrum import read_spectrum, Spectrum, read_operation, Operation
from .read_utils import EntryType, get_entry_type
from .charge_curve import ChargeCurve, charge_curve_from_tuples
from .charge_curve_spline import ChargeCurveSpline
from .plot_utils import autoscale_turned_off
from .charge_reference import ChargeReference

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
        splines[reference_name] = (ChargeCurveSpline( 
            charge_curve_slices[reference_name],
            s[0]))
        
        # make all other splines
        for index, name in enumerate(peak_names):
            charge_curve = self.get_charge_curve(name)
            charge_curve_slices[name] = charge_curve.slice(
                slice_start, slice_end)
            splines[name] = (ChargeCurveSpline(
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
                reference_splines[name] = (ChargeCurveSpline(
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
        # slice background and edge regions
        edge_data = self.charge_corrected_valence_band.slice(
            edge_range[0], edge_range[1])
        background_data = self.charge_corrected_valence_band.slice(
            background_range[0], background_range[1])
        # fit background region
        X_background = np.array([[1]*len(background_data.counts), background_data.eV]).T
        fit_background = sp.optimize.lsq_linear(X_background, background_data.counts)
        # fit edge region
        X_edge = np.array([[1]*len(edge_data.counts), edge_data.eV]).T # I have no goddamn idea what this does
        fit_edge = sp.optimize.lsq_linear(X_edge, edge_data.counts)
        # find valence band maximum
        VBM = ((fit_edge.x[0]-fit_background.x[0])
              /(fit_background.x[1]-fit_edge.x[1]))
        VBM_with_broadening = VBM - instrumental_broadening / 2
        # plot result
        fig, ax = plt.subplots()
        self.charge_corrected_valence_band.plot(color = "tab:blue")
        with autoscale_turned_off(ax):
            # background linear fit
            ax.plot(self.charge_corrected_valence_band.eV,
                    fit_background.x[0]
                  + self.charge_corrected_valence_band.eV * fit_background.x[1],
                     color = 'tab:green', linestyle = 'dashed')
            # edge linear fit
            ax.plot(self.charge_corrected_valence_band.eV,
                    fit_edge.x[0]
                  + self.charge_corrected_valence_band.eV * fit_edge.x[1],
                    color = 'tab:green', linestyle = 'dashed')
        # background chosen points
        ax.scatter(min(background_data.eV),
            fit_background.x[0] + min(background_data.eV) * fit_background.x[1],
            color = 'tab:green', zorder = 3)
        ax.scatter(max(background_data.eV),
            fit_background.x[0] + max(background_data.eV) * fit_background.x[1],
            color = 'tab:green', zorder = 3)
        # edge chosen points
        ax.scatter(min(edge_data.eV),
            fit_edge.x[0] + min(edge_data.eV) * fit_edge.x[1],
            color = 'tab:green', zorder = 3)
        ax.scatter(max(edge_data.eV),
            fit_edge.x[0] + max(edge_data.eV) * fit_edge.x[1],
            color = 'tab:green', zorder = 3)
        # valence band maximum
        ax.scatter(
            VBM_with_broadening,
            fit_background.x[0] + VBM_with_broadening * fit_background.x[1],
            color = 'tab:red', zorder = 3)  
        ax.vlines(
            VBM,
            fit_edge.x[0]+min(edge_data.eV)*fit_edge.x[1],
            fit_background.x[0]+background_data.eV[0]*fit_background.x[1],
            color = 'tab:red', linestyle = 'dashed')
        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Counts')
        print(f'Valence Band Maximum: {VBM} eV')
        return
    
    def find_fermi_edge(self, smoothing_factor, guess):
        # spline fit
        x = self.charge_corrected_valence_band.eV[::-1]
        y = self.charge_corrected_valence_band.counts[::-1]
        tck = sp.interpolate.splrep(x, y, s=smoothing_factor)
        y_spline = sp.interpolate.BSpline(*tck)(x)
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(15, 5.5)
        # plot data/spline to two panes, zoom in to the second pane
        for ax in axs.flatten():
            ax.plot(x, y, color = 'tab:blue')
            ax.plot(x, y_spline, color = 'tab:orange') 
            ax.vlines(guess, y.min(), y.max(), color = 'red', linestyle = 'dashed')
            ax.invert_xaxis()
        axs[1].set_xlim(guess+2, guess-2)
        fig.suptitle('spline fit to charge-corrected data')
        
        # first derivative
        yp = np.diff(y_spline)/np.diff(x)
        xp = (x[:-1] + x[1:]) / 2 
        # plot first derivative
        fig2, axs2 = plt.subplots(1, 2)
        fig2.set_size_inches(15, 5.5)
        for ax in axs2.flatten():
            ax.plot(xp, yp, color = 'tab:blue')
            ax.vlines(guess, yp.min(), yp.max(), color = 'red', linestyle = 'dashed')
            ax.invert_xaxis()
        axs2[1].set_xlim(guess+2, guess-2)
        fig2.suptitle('first derivative')

        # second derivative
        ypp = np.diff(yp)/np.diff(xp)
        xpp = (xp[:-1] + xp[1:]) / 2 
        # plot second derivative
        fig3, axs3 = plt.subplots(1, 2)
        fig3.set_size_inches(15, 5.5)
        for ax in axs3.flatten():
            ax.plot(xpp, ypp, color = 'tab:blue')
            ax.hlines(0, xpp.min(), xpp.max(), color = 'red', linestyle = 'dashed')
            ax.vlines(guess, ypp.min(), ypp.max(), color = 'red', linestyle = 'dashed')
            ax.invert_xaxis()
        axs3[1].set_xlim(guess+2, guess-2)
        axs3[1].set_ylim(ypp.min()/2, ypp.max()/2)
        fig3.suptitle('second derivative')

def read_datafile(path: str) -> DataFile:
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
datafile = read_datafile
print("data processed with plnt_xps version 0.1.0")
print("our crew is replaceable, your package isn't!")

#Processing Laborsaving iNterpolative Tools for XPS

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy



from contextlib import contextmanager
@contextmanager
def autoscale_turned_off(ax=None):
  ax = ax or plt.gca()
  lims = [ax.get_xlim(), ax.get_ylim()]
  yield
  ax.set_xlim(*lims[0])
  ax.set_ylim(*lims[1])




class datafile:
    
    def __init__(self, filename, hv = 1253.64, work_function = 4.543, use_parabola_fit = False):
        
        spectrum_names = []
        spectrum_comments = []
        spectrum_counts = []
        spectrum_eV = []
        spectrum_times = []
        
        operation_names = []
        operation_counts = []
        operation_eV = []
        operation_parents = []
        operation_centers = []
        operation_center_uncertainties = []
        
        
        with open(filename, 'r') as f:

            searching = True
            while searching:
                for n in range(0, 1000):
                    if n == 999:
                        searching = False
                    newline = f.readline()
                    
                    if 'Result:' in newline:
                        while True:
                            newline = f.readline()
                            
                            if len(newline.split()) == 1:
                                break
                            if use_parabola_fit:
                                if 'Parameter: "Peak (x)"' in newline:
                                    operation_centers.append(float(newline.split()[5])) #these are already in binding energy
                                    operation_center_uncertainties.append(float(newline.split()[8])) 
                            else:
                                if '"Center"' in newline:
                                    operation_centers.append(float(newline.split()[4])) #these are already in binding energy
                                if '"Peak 1/C/Value"' in newline:
                                    operation_centers.append(hv - float(newline.split()[5])) #these are in kinetic energy and need to be converted
                    
                    if 'Result Name:' in newline:
                        newline = newline.split()
                        operation_names.append(''.join(newline[3:]))
                        operation_parents.append(len(spectrum_names)-1)
                        operation_counts.append([])
                        operation_eV.append([])

                        for m in range(0, 5):
                            f.readline()
                        while True:
                            try:
                                newline = f.readline().split()
                                operation_eV[-1].append(float(newline[0]))
                                operation_counts[-1].append(float(newline[1]))
                            except IndexError:
                                break
                        break
                    elif 'Region:' in newline:
                        newline = newline.split()
                        spectrum_names.append(' '.join(newline[2:]))
                        spectrum_counts.append([])
                        spectrum_eV.append([])
                        
                        for m in range(0, 27):
                            newline = f.readline()
                            if 'Comment:' in newline:
                                spectrum_comments.append(' '.join(newline.split()[2:]))
                            if 'Date:' in newline and m < 5:
                                spectrum_times.append(newline.split()[4])
                        while True:
                            try:
                                newline = f.readline().split()
                                spectrum_eV[-1].append(float(newline[0]))
                                spectrum_counts[-1].append(float(newline[1]))
                            except IndexError:
                                break
                        break

            f.close()
        
        #convert times to minutes
        for index, time in enumerate(spectrum_times):
            time_split = time.split(':')
            minutes = float(time_split[0])*60 + float(time_split[1]) + float(time_split[2])/60
            spectrum_times[index] = minutes
            
        
        self.spectrum_names = spectrum_names
        self.spectrum_counts = spectrum_counts
        self.spectrum_eV = spectrum_eV
        self.spectrum_comments = spectrum_comments
        self.spectrum_times = spectrum_times
        
        self.operation_names = operation_names
        self.operation_counts = operation_counts
        self.operation_eV = operation_eV
        self.operation_parents = operation_parents
        self.operation_centers = operation_centers
        self.operation_center_uncertainties = operation_center_uncertainties
        
        self.start_time = self.spectra[0].time
        
    def list_spectra(self):
        ljust_length = 0
        for name in self.spectrum_names:
            if len(name) > ljust_length:
                ljust_length = len(name)
        for index, name in enumerate(self.spectrum_names):
            print(f"{index}:\t{name.ljust(ljust_length, ' ')}    {self.spectra[index].time-self.start_time:.2f}    {self.spectrum_comments[index]}")
            
            for Index, Operation in enumerate(self.operations):
                if index == Operation.parent:
                    print(f"\t\t{Operation.name} (#{Index})")
            
    
    @property
    def spectra(self):
        Spectra = []
        for index, Spectrum in enumerate(self.spectrum_names):
            Spectra.append(spectrum(self, index))
        return Spectra    
    
    @property
    def operations(self):
        Operations = []
        for index, Operation in enumerate(self.operation_names):
            Operations.append(operation(self, index))
        return Operations
    
    def organize_charging_kinetics(self, region_name):
        region_times = []
        region_centers = []
        region_error_bars = []

        #for index, Spectrum in enumerate(self.spectra):
        for index, Spectrum in enumerate(self.spectra_with_peak_location):
            if Spectrum.name == region_name:
                region_times.append(Spectrum.time)
                region_centers.append(self.operation_centers[index])
                region_error_bars.append(self.operation_center_uncertainties[index])

        start_time = self.spectrum_times[0]

        region_times = np.array(region_times)
        region_times = region_times - start_time

        return region_times, region_centers, region_error_bars
    
    @property
    def spectra_with_peak_location(self):
        spectrum_indices = [x.parent for x in self.operations if x.name == "PeakLocation"]
        
        result = [self.spectra[index] for index in spectrum_indices]
        
        return result
    
    def spectra_with_name(self, name):
        return [x for x in self.spectra if x.name == name]
    
    
    #this function is intended for samples with Au to get the charge-corrected binding energies of core photoemission peaks
    def get_charge_referenced_peak_positions(self, time_slice, s = [0.02, 0.02, 0.02], Au_name = 'Au 4f', peaks = ['O 1s', 'Ti 2p'], plot_result = False, pub_result = False):
        if len(s) != len(peaks) + 1:
            raise IndexError('s value needed for each peak type (including gold)')
        
        self.splines = []
        
        
        #these full datasets are just for plotting purposes, they make it easier to see you've got the slice you actually want
        peak_times = []
        peak_centers = []
        peak_uncertainties = []
        
        
        #get times, centers
        Au_times, Au_centers, Au_uncertainties = self.organize_charging_kinetics(Au_name)
        #store full data
        peak_times.append(copy.copy(Au_times))
        peak_centers.append(copy.copy(Au_centers))
        peak_uncertainties.append(copy.copy(Au_uncertainties))
        #apply slice
        Au_centers = Au_centers[np.argmax(Au_times>time_slice[0]):np.argmax(Au_times>time_slice[1])]
        Au_uncertainties = Au_uncertainties[np.argmax(Au_times>time_slice[0]):np.argmax(Au_times>time_slice[1])]
        Au_times = Au_times[np.argmax(Au_times>time_slice[0]):np.argmax(Au_times>time_slice[1])]
        #make the gold spline
        self.splines.append(charge_reference_spline(Au_name, Au_times, Au_centers, Au_uncertainties, s[0]))
        
        
        
        #make all other splines
        for index, name in enumerate(peaks):
            #get times, centers
            times, centers, uncertainties = self.organize_charging_kinetics(name)
            #store full datasets
            peak_times.append(copy.copy(times))
            peak_centers.append(copy.copy(centers))
            peak_uncertainties.append(copy.copy(uncertainties))
            
            #apply slice
            centers = centers[np.argmax(times>time_slice[0]):np.argmax(times>time_slice[1])]
            uncertainties = uncertainties[np.argmax(times>time_slice[0]):np.argmax(times>time_slice[1])]
            times = times[np.argmax(times>time_slice[0]):np.argmax(times>time_slice[1])]
            #make spline
            self.splines.append(charge_reference_spline(name, times, centers, uncertainties, s[index+1]))
            
        
        #plot spline fits
        if plot_result:
            fig, axs = plt.subplots(1, len(s))
            #plot gold
            #axs[0].errorbar(self.splines[0].times, self.splines[0].centers, self.splines[0].uncertainties, ecolor = 'navy', capsize = 2.5, capthick = 1.5)
            
            axs[0].plot(self.splines[0].t, self.splines[0].spline, label = Au_name, linestyle = 'dashed', color = 'tab:orange')
            axs[0].scatter(self.splines[0].times, self.splines[0].centers)
            
            axs[0].plot(peak_times[0], peak_centers[0], color = 'tab:blue')
            axs[0].scatter(peak_times[0], peak_centers[0], facecolors='none', edgecolors='r')
            
            axs[0].set_ylim(np.min(self.splines[0].centers)-0.05, np.max(self.splines[0].centers)+0.05)
            axs[0].set_xlim(np.min(self.splines[0].t)-20, np.max(self.splines[0].t)+20)
            axs[0].set_title(Au_name)
            axs[0].set_ylabel('Binding Energy (eV)')
            fig.supxlabel('Time (minutes)')
            
            
            
            #plot others
            for index, spline in enumerate(self.splines[1:]):
                
                #axs[index+1].errorbar(spline.times, spline.centers, spline.uncertainties, ecolor = 'navy', capsize = 2.5, capthick = 1.5)

                axs[index+1].plot(spline.t, spline.spline, label = peaks[index], linestyle = 'dashed', color = 'tab:orange')
                axs[index+1].scatter(spline.times, spline.centers)
                
                axs[index+1].plot(peak_times[index+1], peak_centers[index+1], color = 'tab:blue')
                axs[index+1].scatter(peak_times[index+1], peak_centers[index+1], facecolors='none', edgecolors='r')
                
                axs[index+1].set_ylim(np.min(spline.centers)-0.05, np.max(spline.centers)+0.05)
                axs[index+1].set_xlim(np.min(spline.t)-20, np.max(spline.t)+20)
                
                axs[index+1].set_title(spline.name)
            fig.set_size_inches(16, 5)
        
        if pub_result:
            fig, axs = plt.subplots(1, len(s))
            #plot gold
            axs[0].errorbar(self.splines[0].times, self.splines[0].centers, self.splines[0].uncertainties, linestyle = '', ecolor = 'navy', capsize = 2.5, capthick = 1.5)
            
            axs[0].plot(self.splines[0].t, self.splines[0].spline, linestyle = 'dashed', color = 'tab:orange', label = 'spline')
            axs[0].scatter(self.splines[0].times, self.splines[0].centers, label = 'data')
            
            #axs[0].plot(peak_times[0], peak_centers[0], color = 'tab:blue')
            #axs[0].scatter(peak_times[0], peak_centers[0], facecolors='none', edgecolors='r')
            
            axs[0].set_ylim(np.min(self.splines[0].centers)-0.05, np.max(self.splines[0].centers)+0.05)
            #axs[0].set_xlim(np.min(self.splines[0].t)-20, np.max(self.splines[0].t)+20)
            axs[0].set_title(Au_name)
            axs[0].set_ylabel('Binding Energy (eV)')
            fig.supxlabel('Time (minutes)')
            axs[0].legend(frameon = False)
            
            
            #plot others
            for index, spline in enumerate(self.splines[1:]):
                
                axs[index+1].errorbar(spline.times, spline.centers, spline.uncertainties, linestyle = '', ecolor = 'navy', capsize = 2.5, capthick = 1.5)

                axs[index+1].plot(spline.t, spline.spline, label = peaks[index], linestyle = 'dashed', color = 'tab:orange')
                axs[index+1].scatter(spline.times, spline.centers)
                
                #axs[index+1].plot(peak_times[index+1], peak_centers[index+1], color = 'tab:blue')
                #axs[index+1].scatter(peak_times[index+1], peak_centers[index+1], facecolors='none', edgecolors='r')
                
                axs[index+1].set_ylim(np.min(spline.centers)-0.05, np.max(spline.centers)+0.05)
                #axs[index+1].set_xlim(np.min(spline.t)-20, np.max(spline.t)+20)
                
                axs[index+1].set_title(spline.name)
            fig.set_size_inches(16, 5)
        
        #find common time domain for all splines
        t_common_min = self.splines[0].times[0]
        for spline in self.splines[1:]:
            if spline.times[0] > t_common_min:
                t_common_min = spline.times[0]
                
        t_common_max = self.splines[0].times[-1]
        for spline in self.splines[1:]:
            if spline.times[-1] < t_common_max:
                t_common_max = spline.times[-1]
                
        t_common = np.arange(t_common_min, t_common_max, 0.01)
        
        #reference splines to the Au 4f 7/2
        charge_correction_curve = 84 - self.splines[0].interpolate(t_common)
        
        charge_corrected_splines = []
        for spline in self.splines[1:]:
            charge_corrected_splines.append(spline.interpolate(t_common) + charge_correction_curve)
        
        #print/store results
        self.charge_corrected_peak_positions = []
        for index, spline in enumerate(self.splines[1:]):
            print(f'{spline.name} peak position: {np.mean(charge_corrected_splines[index]):.4f} +- {np.std(charge_corrected_splines[index]):.4f}')
            self.charge_corrected_peak_positions.append(np.mean(charge_corrected_splines[index]))
        
        return
    
    #this function is intended to be used for samples without Au to reference valence band measurements to core photoemission lines, provided the "true" binding energy of those lines is known
    def charge_reference_valence_band_spectra(self, time_slice, reference_spectrum, s = [0.02, 0.02], reference_peaks = ['O 1s', 'Ti 2p'], valence_band_name = "Valence Band", save_figures = ""):
        if len(s) != len(reference_peaks):
            raise IndexError('s value needed for each peak type (including gold)')
        reference_peak_positions = reference_spectrum.charge_corrected_peak_positions
        
        self.reference_splines = []
        
        #make splines for reference peaks
        for index, name in enumerate(reference_peaks):
            #get times, centers
            times, centers, uncertainties = self.organize_charging_kinetics(name)

            #apply slice
            centers = centers[np.argmax(times>time_slice[0]):np.argmax(times>time_slice[1])+1]
            uncertainties = uncertainties[np.argmax(times>time_slice[0]):np.argmax(times>time_slice[1])+1]
            times = times[np.argmax(times>time_slice[0]):np.argmax(times>time_slice[1])+1]
            #make spline
            self.reference_splines.append(charge_reference_spline(name, times, centers, uncertainties, s[index]))
            
        #find a common time domain for the splines
        t_common_min = self.reference_splines[0].times[0]
        t_common_max = self.reference_splines[0].times[-1]
        if len(reference_peaks) > 1:
            for spline in self.reference_splines[1:]:
                if spline.times[0] > t_common_min:
                    t_common_min = spline.times[0]
                if spline.times[-1] < t_common_max:
                    t_common_max = spline.times[-1]
        
        t_common = np.arange(t_common_min, t_common_max+0.01, 0.01)
        
        #define charge correction splines as functions
        def charge_correction_splines(t):
            result = []
            for index, spline in enumerate(self.reference_splines):
                result.append(reference_peak_positions[index] - spline.interpolate(t))
            
            return np.array(result)
        
        def mean_charge_correction_spline(t):
            return np.mean(charge_correction_splines(t))
        
        #calculate a charge-correction curve for each spline (just for the sake of plotting it)
        charge_correction_curves = []
        #for index, spline in enumerate(self.reference_splines):
        #    charge_correction_curves.append(reference_peak_positions[index] - spline.interpolate(t_common))
        for t in t_common:
            values = charge_correction_splines(t)
            charge_correction_curves.append(values)
        charge_correction_curves = np.transpose(np.array(charge_correction_curves))
        
        #calculate a mean charge-correction curve (just for the sake of plotting it)
        mean_charge_correction_curve = []
        for t in t_common:
            mean_charge_correction_curve.append(mean_charge_correction_spline(t))
        mean_charge_correction_curve = np.array(mean_charge_correction_curve)
        
        
        #plot splines
        fig, axs = plt.subplots(1, len(s))
        
        for index, spline in enumerate(self.reference_splines):
                
                #axs[index+1].errorbar(spline.times, spline.centers, spline.uncertainties, ecolor = 'navy', capsize = 2.5, capthick = 1.5)

                axs[index].plot(spline.t, spline.spline, linestyle = 'dashed', color = 'tab:orange', label = 'spline')
                axs[index].scatter(spline.times, spline.centers, label = 'data')
                axs[index].errorbar(spline.times, spline.centers, yerr=spline.uncertainties, color = 'tab:blue', ecolor = 'navy', capsize = 2.5, capthick = 1.5)
                
                #axs[index].plot(peak_times[index+1], peak_centers[index+1], color = 'tab:blue')
                #axs[index].scatter(peak_times[index+1], peak_centers[index+1], facecolors='none', edgecolors='r')
                
                axs[index].set_ylim(np.min(spline.centers)-0.05, np.max(spline.centers)+0.05)
                axs[index].set_xlim(np.min(spline.t)-20, np.max(spline.t)+20)
                
                axs[index].set_title(spline.name)
                axs[index].set_xlabel('time (minutes)')
                
        axs[0].legend()
        axs[0].set_ylabel('binding energy (eV)')
        fig.suptitle('spline fits')
        fig.set_size_inches(16, 5)
        
        #plot charge correction curves
        fig2, ax2 = plt.subplots(1, 2)
        for index, curve in enumerate(charge_correction_curves):
            ax2[0].plot(t_common, curve, label = reference_peaks[index])
        ax2[0].plot(t_common, mean_charge_correction_curve, label = 'mean')
        ax2[0].legend()
        ax2[0].set_title('charge correction curves')
        ax2[0].set_xlabel('time (minutes)')
        ax2[0].set_ylabel('binding energy shift (eV)')
        fig2.set_size_inches(16, 5)
        
        #collect all valence band spectra
        valence_band_spectra = [x for x in self.spectra if x.name == valence_band_name and (time_slice[0] <= x.time - self.start_time <= time_slice[1])]
        print(f'time slice excludes {len([x for x in self.spectra if x.name == valence_band_name])-len(valence_band_spectra)} of {len([x for x in self.spectra if x.name == valence_band_name])} valence band spectra')
        
        
        
        #get the corresponding charge correction shift to each valence band spectrum
        valence_band_charge_corrections = []
        for Spectrum in valence_band_spectra:
            valence_band_charge_corrections.append(mean_charge_correction_spline(Spectrum.time-self.start_time))
        valence_band_charge_corrections = np.array(valence_band_charge_corrections)
        
        
        
        #define a common set of points that our valence band spectra will use as x-values, trimmed down by 0.5 eV on either side
        #window_cutoff=np.abs(mean_charge_correction_curve).max()*1.2
        #eV_window = np.arange(valence_band_spectra[0].eV[-1]+window_cutoff, valence_band_spectra[0].eV[0]+0.01-window_cutoff, 0.01)[::-1]
        
        window_cutoff = [0, 0]
        window_cutoff[0] = valence_band_charge_corrections.max()
        window_cutoff[1] = valence_band_charge_corrections.min()
        eV_window = np.arange(valence_band_spectra[0].eV[-1]+window_cutoff[0], valence_band_spectra[0].eV[0]+window_cutoff[1], 0.01)[::-1]
        
        #create a list of all shifted valence band spectra
        shifted_valence_band_spectra = []
        for index, Spectrum in enumerate(valence_band_spectra):
            shifted_valence_band_spectra.append([Spectrum.eV + valence_band_charge_corrections[index], np.array(Spectrum.counts)])
            #print(f'applying shift of {valence_band_charge_corrections[index]}')
        
        #created a list of linear interpolation functions which correspond to the shifted valence band spectra
        shifted_valence_band_interpolations = []
        for Spectrum in shifted_valence_band_spectra:
            shifted_valence_band_interpolations.append(sp.interpolate.interp1d(Spectrum[0], Spectrum[1]))

        
        #add them all up
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
        
        #store the data
        self.charge_corrected_valence_band_eV = eV_window
        self.charge_corrected_valence_band_counts = shifted_valence_band_sum
        
        #plot the sum of all valence band spectra, with and without shifting
        valence_band_spectrum = np.array(valence_band_spectra[0].counts)
        for Spectrum in valence_band_spectra[1:]:
            valence_band_spectrum += np.array(Spectrum.counts)
        
        fig3, ax3 = plt.subplots(1, 2)
        fig3.set_size_inches(16, 5)
        #fig3.suptitle('USR 450 valence band')
        
        ax3[0].plot(eV_window, shifted_valence_band_sum, label = 'charge-corrected')
        ax3[0].plot(valence_band_spectra[0].eV, valence_band_spectrum, label = 'no shift applied', linestyle = 'dashed')
        ax3[0].legend()
        ax3[0].invert_xaxis()
        ax3[0].set_xlabel("binding energy (eV)")
        ax3[0].set_ylabel("counts")
        
        ax3[1].plot(eV_window, shifted_valence_band_sum, label = 'charge-corrected')
        ax3[1].plot(valence_band_spectra[0].eV, valence_band_spectrum, label = 'no shift applied', linestyle = 'dashed')
        ax3[1].legend()
        ax3[1].set_xlim(2, 6)
        ax3[1].invert_xaxis()
        ax3[1].set_xlabel("binding energy (eV)")
        
        if len(save_figures) > 0:
            fig.savefig(f'{save_figures} splines.svg')
            fig2.savefig(f'{save_figures} charge correction curves.svg')
            fig3.savefig(f'{save_figures} valence band.svg')
        
        
        return
    
    def find_fermi_edge(self, smoothing_factor, guess):
        
        #spline fit:
        x = self.charge_corrected_valence_band_eV[::-1]
        y = self.charge_corrected_valence_band_counts[::-1]

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

    def find_fermi_edge_linfit(self, background_range, edge_range, instrumental_broadening = 0.8):

        background_min = np.amax(np.where(self.charge_corrected_valence_band_eV > background_range[1]))
        background_max = np.amax(np.where(self.charge_corrected_valence_band_eV > background_range[0]))

        edge_min = np.amax(np.where(self.charge_corrected_valence_band_eV > edge_range[1]))
        edge_max = np.amax(np.where(self.charge_corrected_valence_band_eV > edge_range[0]))

        y_background = self.charge_corrected_valence_band_counts[background_min:background_max]
        x_background = self.charge_corrected_valence_band_eV[background_min:background_max]
        X_background = np.array([[1]*len(y_background), x_background]).T
        fit_background = sp.optimize.lsq_linear(X_background, y_background)

        y_edge = self.charge_corrected_valence_band_counts[edge_min:edge_max]
        x_edge = self.charge_corrected_valence_band_eV[edge_min:edge_max]
        X_edge = np.array([[1]*len(y_edge), x_edge]).T
        fit_edge = sp.optimize.lsq_linear(X_edge, y_edge)

        fix, ax = plt.subplots()

        ax.plot(self.charge_corrected_valence_band_eV, self.charge_corrected_valence_band_counts, color = 'tab:blue')

        with autoscale_turned_off(ax):
            ax.plot(self.charge_corrected_valence_band_eV, fit_background.x[0]+self.charge_corrected_valence_band_eV*fit_background.x[1], color = 'tab:green', linestyle = 'dashed')
            ax.plot(self.charge_corrected_valence_band_eV, fit_edge.x[0]+self.charge_corrected_valence_band_eV*fit_edge.x[1], color = 'tab:green', linestyle = 'dashed')
        
        
        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Counts')
        VBM_with_broadening = (fit_edge.x[0]-fit_background.x[0])/(fit_background.x[1]-fit_edge.x[1])
        VBM = VBM_with_broadening + instrumental_broadening/2
        
        
        ax.scatter(self.charge_corrected_valence_band_eV[background_min], fit_background.x[0]+self.charge_corrected_valence_band_eV[background_min]*fit_background.x[1], color = 'tab:green', zorder = 3)
        ax.scatter(self.charge_corrected_valence_band_eV[background_max], fit_background.x[0]+self.charge_corrected_valence_band_eV[background_max]*fit_background.x[1], color = 'tab:green', zorder = 3)
        
        ax.scatter(self.charge_corrected_valence_band_eV[edge_min], fit_edge.x[0]+self.charge_corrected_valence_band_eV[edge_min]*fit_edge.x[1], color = 'tab:green', zorder = 3)
        ax.scatter(self.charge_corrected_valence_band_eV[edge_max], fit_edge.x[0]+self.charge_corrected_valence_band_eV[edge_max]*fit_edge.x[1], color = 'tab:green', zorder = 3)
        
        ax.scatter(VBM_with_broadening, fit_background.x[0]+VBM_with_broadening*fit_background.x[1], color = 'tab:red', zorder = 3)
        
        ax.vlines(VBM, fit_edge.x[0]+self.charge_corrected_valence_band_eV[edge_min]*fit_edge.x[1], fit_background.x[0]+self.charge_corrected_valence_band_eV[background_max]*fit_background.x[1], color = 'tab:red', linestyle = 'dashed')
        
        print(f'Valence Band Maximum: {VBM} eV')
    
    #@property
    #def fermi_level(self): #calculates the fermi level via the second derivative method


class charge_reference_spline:
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
    
    
class spectrum:
    def __init__(self, data_file, index):
        self.counts = data_file.spectrum_counts[index]
        self.eV = data_file.spectrum_eV[index]
        self.name = data_file.spectrum_names[index]
        self.comment = data_file.spectrum_comments[index]
        self.time = data_file.spectrum_times[index]
        
    def plot(self, ax = None, **kwargs):
        print(f'{self.name}    {self.comment}')
        if type(ax) == type(None):
            plt.plot(self.eV, self.counts, **kwargs)
            plt.gca().xaxis.set_inverted(True)
        else:
            ax.plot(self.eV, self.counts, **kwargs)
            ax.xaxis.set_inverted(True)
    
    
class operation:
    def __init__(self, data_file, index):
        self.counts = data_file.operation_counts[index]
        self.eV = data_file.operation_eV[index]
        self.name = data_file.operation_names[index]
        self.parent = data_file.operation_parents[index]
        self.parent_name = data_file.spectrum_names[self.parent]
        
    def plot(self, ax = None, **kwargs):
        if type(ax) == type(None):
            plt.plot(self.eV, self.counts, **kwargs)
            plt.gca().xaxis.set_inverted(True)
        else:
            ax.plot(self.eV, self.counts, **kwargs)
            ax.xaxis.set_inverted(True)

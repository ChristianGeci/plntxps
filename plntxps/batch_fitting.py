from glob import glob
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .core import read_datafile
from os import mkdir
from lmfit.model import save_modelresult
import pandas as pd
from .logger import Logger, NullLogger
from .fitting import *

@dataclass
class XpsBatchFit:
    experiment_table: pd.DataFrame
    region_table: pd.DataFrame
    fit_table: pd.DataFrame
    peak_tables: dict[str, pd.DataFrame]
    satellites: dict

    def slice_requested(self, region):
        slice_bounds_string = (
            self.region_table.query('region == @region')
            ['slice'].item()
        )
        return not pd.isna(slice_bounds_string)
    def get_slice_bounds(self, region):
        slice_bounds_string = (
            self.region_table.query('region == @region')
            ['slice'].item()
        )
        parsed_string = slice_bounds_string.split('-')
        lower_bound = float(parsed_string[0])
        upper_bound = float(parsed_string[1])
        return lower_bound, upper_bound
    def get_spectrum(self, experiment, region):
        spectrum_index = (
            self.fit_table.query('label == @experiment')
            [region].item()
        )
        if pd.isna(spectrum_index):
            return None
        spectrum = (
            self.experiment_table.query('label == @experiment')
            ['data'].item().spectra[int(spectrum_index)]
        )
        if self.slice_requested(region):
            slice_lower_bound, slice_upper_bound = self.get_slice_bounds(region)
            spectrum = spectrum.slice(slice_lower_bound, slice_upper_bound)
        return spectrum
    def get_params_path(self, region):
        params_path = (
            self.region_table.query('region == @region')
            ['params file'].item()
        )
        return params_path
    def get_guess_shirley(self, region):
        result = (
            self.region_table.query('region == @region')
            ['guess shirley'].item()
        )
        return result
    def get_bg_type(self, region):
        result = (
            self.region_table.query('region == @region')
            ['background type'].item()
        )
        return result
    def region_should_be_fit(self, region):
        result = (
            self.region_table.query('region == @region')
            ['do fit'].item()
        )
        return result
    def experiment_should_be_fit(self, experiment):
        result = (
            self.fit_table.query('label == @experiment')
            ['do fit'].item()
        )
        return result
    @property
    def all_regions(self):
        return list(self.region_table['region'])
    @property
    def all_fitted_experiments(self):
        return list(self.fit_table['label'])

    def check_guess(self, experiment, region):
        peak_table = self.peak_tables[region]
        params_path = self.get_params_path(region)
        bg_type = self.get_bg_type(region)
        guess_shirley = self.get_guess_shirley(region)
        spectrum = self.get_spectrum(experiment, region)
        print(f"resolution: {spectrum.info.resolution}")
        fit_model = setup_fit_model(peak_table, bg_type, self.satellites)
        plot_initial_guess(
            fit_model, params_path, spectrum.eV, spectrum.counts, guess_shirley)
        plt.show()
        fit = do_fit(
            spectrum.eV, spectrum.counts, fit_model, params_path, guess_shirley)
        plot_fit_result(spectrum.eV, spectrum.counts, fit, self.satellites)
        return fit

    def do_batch_fit(self, output_dir, logger: Logger = NullLogger()):
        try: mkdir(output_dir)
        except FileExistsError: pass
        for region in self.all_regions:
            if not self.region_should_be_fit(region): continue
            try: mkdir(f"{output_dir}/{region}")
            except FileExistsError: pass
            logger.log(f"starting fit of {region}")
            peak_table = self.peak_tables[region]
            params_path = self.get_params_path(region)
            bg_type = self.get_bg_type(region)
            guess_shirley = self.get_guess_shirley(region)
            for experiment in self.all_fitted_experiments:
                if not self.experiment_should_be_fit(experiment): continue
                spectrum = self.get_spectrum(experiment, region)
                if type(spectrum) == type(None): continue
                fit_model = setup_fit_model(peak_table, bg_type, self.satellites)
                fit = do_fit(
                    spectrum.eV, spectrum.counts, fit_model,
                    params_path, guess_shirley)
                plot_fit_result(spectrum.eV, spectrum.counts, fit, self.satellites)
                plt.savefig(f"{output_dir}/{region}/{experiment}.svg")
                plt.close()
                save_modelresult(fit, f"{output_dir}/{region}/{experiment}.json")

def xps_batch_fit(experiment_table_filepath, region_table_filepath):
    experiment_table = pd.read_csv(experiment_table_filepath, sep = '\t')
    region_table = pd.read_csv(region_table_filepath, sep = '\t')

def make_blank_experiment_table(data_directory_path, experiment_table_path):
    # todo: prevent overwriting
    data_paths = glob(f"{data_directory_path}*.xy")
    with open(experiment_table_path, 'w') as f:
        f.write("filepath\tlabel\n")
        for path in data_paths:
            print(path) # debug
            f.write(f"{path}\t\n")
        f.close()
    return
def read_experiment_table(experiment_table_filepath):
    result = pd.read_csv(experiment_table_filepath, sep = '\t')
    datafiles = []
    for index, row in result.iterrows():
        datafiles.append(read_datafile(row['filepath']))
    result['data'] = datafiles
    return result

def make_blank_region_table(filepath):
    # todo: prevent overwriting
    with open(filepath, 'w') as f:
        f.write(
            'region'
            '\tparams file'
            '\tpeaks file'
            '\tslice'
            '\tdo fit'
            '\tbackground type'
            '\tguess shirley'
            )
        f.close()
    return
def fill_out_region_table(filepath, peaks_dir_path, params_dir_path):
    region_table = pd.read_csv(filepath, sep = '\t')
    peak_paths = []
    params_paths = []
    for index, row in region_table.iterrows():
        peak_path = f"{peaks_dir_path}/{row['region']}.csv"
        params_path = f"{params_dir_path}/{row['region']}.csv"
        peak_paths.append(peak_path)
        params_paths.append(params_path)
    region_table['params file'] = params_paths
    region_table['peaks file'] = peak_paths
    region_table.to_csv(filepath, sep = '\t', index = False)

def make_empty_fit_table(experiment_table, region_table, filepath):
    # todo: prevent overwriting
    fit_table = pd.DataFrame()
    fit_table['label'] = experiment_table['label']
    for region in region_table['region']:
        fit_table[region] = None
    fit_table['do fit'] = None
    fit_table.to_csv(filepath, sep = '\t', index = False)
    return

def make_empty_peak_tables(region_table, directory_path):
    empty_table = (
        "peak name"
        "\thas satellites"
        "\tpeak shape"
    )
    try: mkdir(directory_path)
    except FileExistsError: pass
    for region in region_table['region']:
        filepath = f"{directory_path}/{region}.csv"
        with open(filepath, 'w') as f:
            f.write(empty_table)
            f.close()
    return

def read_peak_tables(region_table):
    result = {}
    for index, row in region_table.iterrows():
        region = row['region']
        filepath = row['peaks file']
        peak_table = pd.read_csv(filepath, sep = '\t')
        result[region] = peak_table
    return result

def make_params_files(region_table, peak_tables, satellites):
    for index, row in region_table.iterrows():
        region = row['region']
        peak_table = peak_tables[region]
        path = row['params file']
        setup_fit_params(peak_table, path, satellites)
    return
#set document(title: [XPS Batch Fitting Guide])

#title()

+ Begin by calling `make_experiment_table()`
  - Pass a directory of xps data files and the name of the output `.csv` file as arguments.
  - The experiment table essentially associates .xy data files with an alias and any additional data the user wants to assign (e.g. calcination temperature, treatment time, etc.)

+ Manually or programatically assign labels to each experiment.
  - These labels will essentially serve as the alias of each experiment.
  - Each label should be unique.
  - If `make_experiment_table()` is called again, it will not clobber the old experiment table.
  - If new datafiles are detected, it should add the new datafiles to the experiment table.

+ Read the experiment table back in

+ Call `make_spectrum_lists()` supplying the experiment table and a target directory as the arguments.
  - This creates human-readable descriptions of the spectra contained in each datafile

+ Take a look at your spectra and consider which regions you'd like to fit.

+ Make a blank region table by calling `make_blank_region_table()`.
 - The region table defines which fitting strategies will be employed, e.g. a fit of the C 1s region, a fit of the Ca 2p region, etc.
+ Fill out the `region` column of the region table manually.
+ Call `fill_out_region_table()`
  - Supply the region table path, the peaks directory path, and the params directory path as arguments.
    - ⚠⚠⚠ COMMENT OUT `fill_out_region_table` AFTER RUNNING IT AS IT CAN CLOBBER THE REGION TABLE ⚠⚠⚠
+ Read in the region table like any other Pandas dataframe (note that the delimiter is `\t`)
+ Make empty peak tables using the region table.
+ Fill out the peak tables.
  - The peak tables allow you to specify which peaks will be included in a fit (e.g. Ag 3d 5/2, Ag 3d 3/2, auger 1, plasmon 1, etc.) and allows the user to specify whether the peaks will have x-ray satellites or not. The user can also specify the peak shape (DS should be the default) although this feature currently is not implemented and is only a dummy variable
+ Read the peak tables

+ Make fit table by calling `make_fit_table`
  - supply the experiment table, region table, and fit table filepath as arguments
  - the fit table essentially associates for each data file which spectrum will be used for a given fitting strategy. Every cell in the table corresponds to the index of the spectrum to be fitted. Blank cells are allowed and will be skipped
+ Manually fill out the fit table
+ Read the fit table as a dataframe.

+ Setup empty overrides. // Move to beginning?
  - you will need a directory to read from
  - These will be used later to triage any fits that went wrong

+ Load the satellites 
+ Make param files
+ Manually fill out / modify param files


+ set up some guinea pig fits to make sure the parameters are reasonable and adjust as necessary
  - the `check_guess` method will output the calculated FWHM of the experimental data's instrumental broadening which can be used with the `calculate_gaussian_sigma` function along with the FWHM of the source (0.7 for Mg, 0.85 for Al) to find the sigma value that the gaussian component of the DS line should be fixed to

+ do the batch fit

+ review the results and setup overrides as needed
  - be sure to disable fits you've already run and are happy with to save time
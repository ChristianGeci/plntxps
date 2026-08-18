#set document(title: [XPS Fitting Guide])

#title()

Begin by calling `make_experiment_table()`, passing a directory of xps data files and the name of the output `.csv` file as arguments.

```
    make_experiment_table('xps spectra/', 'experiment table.csv')
```

Manually or programatically assign labels to each experiment.
These labels will essentially serve as the alias of each experiment.
Each label should be unique.
If `make_experiment_table()` is called again, it will not clobber the old experiment table.
If new datafiles are detected, it should add the new datafiles to the experiment table.

Next, call `make_spectrum_lists()` supplying the experiment table and a target directory as the arguments.

```
    make_spectrum_lists(experiment_table, 'spectrum lists')
```

Take a look at your spectra and consider which regions you'd like to fit.
Next, make a blank region table by calling `make_blank_region_table()`, then fill out the `region` column of the region table manually.

```
    make_blank_region_table('region table.csv')
```

After you've filled out the region names, call `fill_out_region_table()`, supplying the region table path, the peak directory path, and the params directory path as arguments.

```
    fill_out_region_table('region table.csv', 'peaks', 'params')
```
⚠⚠⚠ COMMENT OUT `fill_out_region_table` AFTER RUNNING IT AS IT CAN CLOBBER THE REGION TABLE ⚠⚠⚠

Read in the region table like any other Pandas dataframe (note that the delimiter is `\t`)
```
    region_table = pd.read_csv('region table.csv', sep = '\t')
```

Make empty peak tables using the region table.
```
    make_empty_peak_tables(region_table, 'peaks')
```

Plot some examples to see what you're looking at, then fill out the peak tables.

Get the satellites and read the peak tables.

Make param files.

Make fit table. // Maybe do this before params files
Then read it in as a dataframe.

Setup empty overrides. // Move to beginning

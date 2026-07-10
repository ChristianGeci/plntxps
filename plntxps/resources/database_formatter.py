import pandas as pd
import re

photoemission_df = pd.read_csv("HandbookXPS.csv", sep = ';')

print(photoemission_df)
formatted_photoemission_df = pd.DataFrame()

def str_to_float(string):
    return float(re.sub(r'[^0-9.]', "", string))

binding_energies = []
for index, row in photoemission_df.iterrows():
    binding_energies.append(str_to_float(row["BindingEnergy"]))

formatted_photoemission_df["Element"] = photoemission_df["AtomicLevel.symbol"]
formatted_photoemission_df["Level"] = photoemission_df["AtomicLevel.level"]
formatted_photoemission_df["Binding Energy"] = binding_energies

print(formatted_photoemission_df)

formatted_photoemission_df.to_csv("HandbookXPS_formatted.csv",
    sep = '\t', index = False)
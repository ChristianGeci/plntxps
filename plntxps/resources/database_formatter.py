import pandas as pd
import re

def str_to_float(string):
    return float(re.sub(r'[^0-9.]', "", string))

# Direct Photoemission

photoemission_df = pd.read_csv("HandbookXPS.csv", sep = ';')
formatted_photoemission_df = pd.DataFrame()

binding_energies = []
for index, row in photoemission_df.iterrows():
    binding_energies.append(str_to_float(row["BindingEnergy"]))

formatted_photoemission_df["Element"] = photoemission_df["AtomicLevel.symbol"]
formatted_photoemission_df["Level"] = photoemission_df["AtomicLevel.level"]
formatted_photoemission_df["Binding Energy"] = binding_energies

formatted_photoemission_df.to_csv("HandbookXPS_formatted.csv",
    sep = '\t', index = False)

# Auger Emission

auger_df = pd.read_csv("HandbookAES.csv", sep = ';')
formatted_auger_df = pd.DataFrame()

kinetic_energies = []
for index, row in auger_df.iterrows():
    kinetic_energies.append(str_to_float(row["KineticEnergy"]))

formatted_auger_df["Element"] = auger_df["AtomicLevel.symbol"]
formatted_auger_df["Level"] = auger_df["AtomicLevel.level"]
formatted_auger_df["Kinetic Energy"] = kinetic_energies

formatted_auger_df.to_csv("HandbookAES_formatted.csv",
    sep = '\t', index = False)

from dataclasses import dataclass
import re

@dataclass
class XpsPeak:
    element: str
    spin_orbital: str
    binding_energy: float # in eV

def read_xps_peaks(filepath):
    result = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]
        f.close()
    for line in lines:
        fields = line.split(';')
        element = re.sub('"', '', fields[0])
        spin_orbital = re.sub('"', '', fields[1])
        binding_energy = float(re.sub(r'[^\d.]+', '', fields[3]))
        result.append(XpsPeak(element, spin_orbital, binding_energy))
    return result

print(read_xps_peaks("HandbookXPS.csv"))
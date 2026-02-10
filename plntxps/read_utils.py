import re
import numpy as np
from enum import Enum

from .peak_location import PeakLocation

class EntryType(Enum):
    SPECTRUM = 0
    OPERATION = 1

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

def get_comment(entry):
    pattern = r"# Comment:.*\n"
    comment_line = re.search(pattern, entry).group()
    unwanted_part = r"# Comment:\s*"
    comment = re.sub(unwanted_part, "", comment_line)
    return comment.strip()

def get_entry_type(entry):
    spectrum_pattern = r"# Region:.*\n"
    operation_pattern = r"# Operation:.*\n"
    if re.search(spectrum_pattern, entry) is not None:
        return EntryType.SPECTRUM
    if re.search(operation_pattern, entry) is not None:
        return EntryType.OPERATION
    
def get_operation_name(entry):
    pattern = r"# Operation:.*\n"
    line = re.search(pattern, entry).group()
    unwanted_part = r"# Operation:\s*"
    name = re.sub(unwanted_part, "", line)
    return name.strip()

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
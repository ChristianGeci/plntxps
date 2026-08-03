from pandas import read_csv
from re import sub

def read_satellite_peaks(path):
    satellites = read_csv(path,
        sep = '\t').to_dict(orient='index')
    def format_satellite_name(name):
        formatted_name = sub(r" ", "_", name)
        formatted_name = sub(r",", "", formatted_name)
        return formatted_name
    for satellite in satellites.values():
        satellite['name'] = format_satellite_name(satellite['name'])
    return satellites
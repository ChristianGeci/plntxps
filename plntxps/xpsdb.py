from importlib.resources import files

# data_path = files('../resources/HandbookXPS.csv')
data_path = files('plntxps.resources').joinpath('HandbookXPS.csv')

xpsdb_txt = data_path.read_text(encoding='utf-8')
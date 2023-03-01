# Script that reads in data generated

import os
import pandas as pd

file_folder = ["data", "essays.csv"]
cwd = os.getcwd()
file_path = os.path.join(cwd, file_folder)

df = pd.read_csv(file_path)
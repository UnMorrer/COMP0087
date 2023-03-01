# Script that reads in data generated

import os
import pandas as pd

cwd = os.getcwd()
file_path = os.path.join(cwd, "data", "essays.csv")

df = pd.read_csv(file_path)

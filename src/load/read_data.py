# Script that reads in data generated

import os
import pandas as pd

def merged_data():
    """
    OS-agnostic function to load merged data
    """
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "data", "essays_raw.csv")

    df = pd.read_csv(file_path, index_col=0)

    return df

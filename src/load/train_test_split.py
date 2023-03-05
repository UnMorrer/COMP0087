# Load train, test and validation dataset using pandas dataframe
import os
import pandas as pd
import read_data as read

df = read.merged_data()

# Train, test, (validation) proportions
train = 0.7
test = 0.2
validation = 1 - train - test # What is left
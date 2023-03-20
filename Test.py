import torch
from torch import nn
from Data_Loader import Data_Loader, TextDataset

#hyperparameters
batch_size = 64


val_path = r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\play\essays_validation.csv'
valloader = Data_Loader(TextDataset(val_path), batch_size=batch_size, shuffle=False)

#test
for idx, batch in enumerate(valloader):
    a = 1
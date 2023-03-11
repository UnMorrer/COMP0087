import torch
from torch import nn
from Data_Loader import Data_Loader, TextDataset
from Models import LSTM
from aa import tokenizer

path = r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\play'
dataloader=Data_Loader(TextDataset(path), batch_size=1, shuffle=False)
network = LSTM(768, 256, 2, 2)#to be changed
number_of_epochs = 100
for epoch in range(number_of_epochs):
    for idx, batch in enumerate(dataloader):
        tokenized_batched = tokenizer(batch)
        break
    break
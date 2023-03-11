import torch
from torch import nn
from torch.optim import Adam
from Data_Loader import Data_Loader, TextDataset
from Models import LSTM
from aa import BERT_Tokenizer

train_path = r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\play'
#hyperparameters
batch_size = 16
tokenizer = BERT_Tokenizer()
dataloader=Data_Loader(TextDataset(train_path), batch_size=batch_size, shuffle=False)
max_number_of_tokens = 750
input_siz = 768
network = LSTM (input_size = max_number_of_tokens, hidden_size = input_siz, num_layers = 2, num_classes = 2)#to be changed
number_of_epochs = 100
optimizer = Adam(network.parameters(), lr=0.001)
criteron = nn.CrossEntropyLoss()

for epoch in range(number_of_epochs):
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        tokenized_batched = tokenizer(batch[1])
        pred = network(tokenized_batched)
        loss = criteron(pred, batch[0])
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} loss: {loss.item()}')
torch.save(network.state_dict(), f'Model.pt')
        
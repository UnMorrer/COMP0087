import torch
from torch import nn
from torch.optim import Adam
from Data_Loader import Data_Loader, TextDataset
from Models import LSTM
from src.tokenization.general_hf_tokenizer import tokenize_input

train_path = r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\play\essays_train.csv'
test_path = r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\play\essays_test.csv'
#hyperparameters
batch_size = 32
trainloader=Data_Loader(TextDataset(train_path), batch_size=batch_size, shuffle=False)
testloader = Data_Loader(TextDataset(test_path), batch_size=batch_size, shuffle=False)
max_number_of_tokens = 750
input_siz = 768
network = LSTM (input_size = max_number_of_tokens, hidden_size = input_siz, num_layers = 2, num_classes = 1)#to be changed
number_of_epochs = 100
optimizer = Adam(network.parameters(), lr=0.001)
criteron = nn.CrossEntropyLoss()

for epoch in range(number_of_epochs):
    cum_loss = 0
    for idx, batch in enumerate(trainloader):
        optimizer.zero_grad()
        tokenized_batched = tokenize_input(text = batch['input_ids'],num_tokens = max_number_of_tokens)
        pred = network(tokenized_batched)
        loss = criteron(pred.squeeze(-1), batch['label'].float())
        loss.backward()
        cum_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch} train loss: {cum_loss.item()/len(trainloader)}')
    #test loss
    test_cum_loss = 0
    for idx, batch in enumerate(testloader):
        tokenized_batched = tokenize_input(text = batch['input_ids'],num_tokens = max_number_of_tokens)
        pred = network(tokenized_batched)
        loss = criteron(pred.squeeze(-1), batch['label'].float())
        test_cum_loss += loss.item()
    print(f'Epoch {epoch} test loss: {test_cum_loss.item()/len(testloader)}')
torch.save(network.state_dict(), f'Model.pt')
        
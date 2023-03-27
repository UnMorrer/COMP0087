import torch
from torch import nn
from torch.optim import Adam, SGD
import os
from Data_Loader import Data_Loader, TextDataset
from Models import LSTM, RNN, Transformer
from src.tokenization.general_hf_tokenizer import tokenize_input
from transformers import BertTokenizer, BertModel
import re
def Train(model):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # train_path = r'.\data\essays_train.csv'
    train_path = r'essays_train.csv'
    test_path = r'essays_test.csv'
    #hyperparameters
    batch_size = 16
    trainloader=Data_Loader(TextDataset(train_path), batch_size=batch_size, shuffle=False)
    testloader=Data_Loader(TextDataset(test_path), batch_size=batch_size, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokennizer_model = BertModel.from_pretrained('bert-base-uncased')
    max_number_of_tokens = 750
    input_siz = 768
    if model == 'LSTM':
        network = LSTM (input_size = max_number_of_tokens, hidden_size = input_siz, num_layers = 2, num_classes = 1, device = device)#to be changed
    elif model == 'RNN':
        network = RNN (input_size = max_number_of_tokens, hidden_size = input_siz, num_layers = 2, num_classes = 1)#to be changed
    elif model == 'Transformer':
        network = Transformer ()#to be changed
    list_of_files = os.listdir()
    weights_files = []
    # for file in list_of_files:
    #     if file.startswith('Model-LSTM-Epoch'):
    #         weights_files.append(file)
    # if weights_files != []:
    #     weights_files = sorted(weights_files, key=lambda s: int(re.search(r'\d+', s).group()))
    #     network.load_state_dict(torch.load(weights_files[-1]))
    #     epoch_min = int(weights_files[-1].split('Epoch')[1].split('.')[0])
    # else:
    #     epoch_min = 0
    epoch_min = 0
    # network = BertModel.from_pretrained('bert-base-uncased')#to be changed
    network.to(device)
    number_of_epochs = 100
    optimizer = SGD(network.parameters(), lr=0.001)
    criteron = nn.CrossEntropyLoss().to(device)

    for epoch in range(epoch_min, number_of_epochs):
        cum_loss = 0
        for idx, batch in enumerate(trainloader):
            optimizer.zero_grad()
            tokenized_batched = tokenize_input(text = batch['input_ids'],num_tokens = max_number_of_tokens, model = tokennizer_model, tokenizer = tokenizer)
            tokenized_batched = tokenized_batched.to(device)
            pred = network(tokenized_batched)
            batch['label'] = batch['label'].to(device)
            loss = criteron(pred.squeeze(-1), batch['label'].float())
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch} train loss: {cum_loss/len(trainloader)}')
        # torch.save(network.state_dict(), f'Model-LSTM-Epoch{epoch}.pt')
        torch.save(network.state_dict(), f'Model.pt')
        test_cum_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                tokenized_batched = tokenize_input(text = batch['input_ids'],num_tokens = max_number_of_tokens, model = tokennizer_model, tokenizer = tokenizer)
                tokenized_batched = tokenized_batched.to(device)
                pred = network(tokenized_batched)
                batch['label'] = batch['label'].to(device)
                loss = criteron(pred.squeeze(-1), batch['label'].float())
                test_cum_loss += loss.item()
            print(f'Epoch {epoch} test loss: {test_cum_loss/len(testloader)}')
if __name__ == '__main__':
    Train('LSTM')
        
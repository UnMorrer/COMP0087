import torch
from torch import nn
from torch.optim import Adam, SGD
from Data_Loader import Data_Loader, TextDataset
from Models import LSTM, RNN, Transformer
from src.tokenization.general_hf_tokenizer import tokenize_input
from transformers import BertTokenizer, BertModel
def Train():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_path = r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\play\essays_train.csv'
    test_path = r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\play\essays_test.csv'
    #hyperparameters
    batch_size = 16
    trainloader=Data_Loader(TextDataset(train_path), batch_size=batch_size, shuffle=False)
    testloader=Data_Loader(TextDataset(test_path), batch_size=batch_size, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokennizer_model = BertModel.from_pretrained('bert-base-uncased')
    max_number_of_tokens = 750
    input_siz = 768
    network = LSTM (input_size = max_number_of_tokens, hidden_size = input_siz, num_layers = 2, num_classes = 1)#to be changed
    # network = RNN (input_size = max_number_of_tokens, hidden_size = input_siz, num_layers = 2, num_classes = 1)#to be changed
    # network = nn.Transformer ()#to be changed
    # network = BertModel.from_pretrained('bert-base-uncased')#to be changed
    number_of_epochs = 100
    optimizer = SGD(network.parameters(), lr=0.001)
    criteron = nn.CrossEntropyLoss().to(device)

    for epoch in range(number_of_epochs):
        cum_loss = 0
        for idx, batch in enumerate(trainloader):
            optimizer.zero_grad()
            tokenized_batched = tokenize_input(text = batch['input_ids'],num_tokens = max_number_of_tokens, model = tokennizer_model, tokenizer = tokenizer)
            pred = network(tokenized_batched)
            loss = criteron(pred.squeeze(-1), batch['label'].float())
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch} train loss: {cum_loss.item()/len(trainloader)}')
        test_cum_loss = 0
        for idx, batch in enumerate(testloader):
            tokenized_batched = tokenize_input(text = batch['input_ids'],num_tokens = max_number_of_tokens)
            pred = network(tokenized_batched)
            loss = criteron(pred.squeeze(-1), batch['label'].float())
            test_cum_loss += loss.item()
        print(f'Epoch {epoch} test loss: {test_cum_loss.item()/len(testloader)}')
    torch.save(network.state_dict(), f'Model.pt')
if __name__ == '__main__':
    Train()
        
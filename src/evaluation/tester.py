# Function to test models on test set & chatGPT output

# General packages
import torch

# Custom packages
import src.load.dataset_hf as load

from Data_Loader import Data_Loader, TextDataset
from Models import LSTM, RNN, Transformer
from src.tokenization.general_hf_tokenizer import tokenize_input
from transformers import BertTokenizer, BertModel

# Settings
file_loc = {
    "test": "data/essays_test.csv",
    "raw": "data/essays_raw.csv"
}

raw, test = load.read_in(
    sample=False,
    data_files=file_loc
)

def model_tester(
        model_obj,
        weights_file,
        dataset,
        batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    pass


def Test(weights_file, model):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # train_path = r'.\data\essays_train.csv'
    validation_path = r'essays_validation.csv'
    #hyperparameters
    batch_size = 16
    validationloader=Data_Loader(TextDataset(validation_path), batch_size=batch_size, shuffle=False)
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
    network.load_state_dict(torch.load(weights_file,map_location=device))
    network.to(device)
    network.eval()
    with torch.no_grad():
        matrix = torch.zeros(2,2)
        for idx, batch in enumerate(validationloader):
            tokenized_batched = tokenize_input(text = batch['input_ids'],num_tokens = max_number_of_tokens, model = tokennizer_model, tokenizer = tokenizer)
            tokenized_batched = tokenized_batched.to(device)
            pred = network(tokenized_batched)
            pred = torch.clamp(pred.round(), 0, 1).squeeze(1).int()
            batch['label'] = batch['label'].to(device)
            for i in range(pred.shape[0]):
                matrix[pred[i]][batch['label'][i].int()] += 1
    #accuracy recall precision f1
    acc = matrix.trace()/matrix.sum()
    recall = matrix[1,1]/matrix[1,:].sum()
    precision = matrix[1,1]/matrix[:,1].sum()
    F1 = 2 * precision*recall/(precision+recall)
    print('Accuracy: ', acc)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1: ', F1)
    print(matrix)

if __name__ == '__main__':
    Test('Model-LSTM-Epoch44.pt', 'LSTM')
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Sigmoid())
    
    def forward(self, enbeded_answer):
        h0 = torch.zeros(self.num_layers, enbeded_answer.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, enbeded_answer.size(0), self.hidden_size)
        # Set initial hidden and cell states 
        lstm_out,_ = self.lstm(enbeded_answer.permute(0,2,1), (h0,c0))
        # Decode the hidden state of the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out
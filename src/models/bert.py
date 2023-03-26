# Fully connected model for classification, based on BERT embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNConnected(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_classes,
                 batch_size,
                 init_hidden_state=None,
                 rnn_layers=3):
        super(FullyConnected, self).__init__()
        # Define the layers of the neural network

        # RNN that runs through the 512-long embedding
        # and gets overall "meaning" out
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=True) # Only works if there is more than 1 RNN layer
        
        # Keep track of RNN hidden states
        if init_hidden_state is None:
            self.h = torch.zeros(2*rnn_layers,
                                 batch_size,
                                 hidden_size)
        else:
            self.f = init_hidden_state

        # Linear/Dense/Fully Connected layer for RNN output
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Define the forward pass of the neural network
        x, self.h = self.rnn(x, self.h) # Run thru RNN + update hidden state
        x = F.relu(self.fc(x))
        return x
    
    def loss(self, outputs, targets):
        # Define the loss function
        loss = nn.BCEWithLogitsLoss()
        return loss(outputs, targets)

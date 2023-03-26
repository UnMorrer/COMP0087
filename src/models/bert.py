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
                 max_tokens_per_essay,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 init_hidden_state=None,
                 rnn_layers=3,
                 bidirectional_rnn=True,
                 loss_reduction="mean",
                 rnn_dropout=0.1): # Proportion for RNN dropout
        super(RNNConnected, self).__init__()
        # Define the layers of the neural network

        # RNN that runs through the 512-long embedding
        # and gets overall "meaning" out
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional_rnn, 
            dropout=rnn_dropout) # Only works if there is more than 1 RNN layer
        
        # Bidirectional RNN
        self.bidirectional_multiplier = 2 if bidirectional_rnn else 1

        # Keep track of RNN hidden states
        if init_hidden_state is None:
            self.h = torch.zeros(self.bidirectional_multiplier*rnn_layers,
                                 batch_size,
                                 hidden_size).to(device)
        else:
            self.f = init_hidden_state.to(device)

        # Linear/Dense/Fully Connected layer for RNN output
        self.fc = nn.Linear(
            # shape [batch_size, self.bidirectional_multiplier*hidden_size*tokens_per_essay]
            self.bidirectional_multiplier*hidden_size*max_tokens_per_essay,
            num_classes)
        
        # Other things to save
        self.loss_reduction = loss_reduction
        
    def forward(self, x):
        # Define the forward pass of the neural network
        x, self.h = self.rnn(x, self.h) # Run thru RNN + update hidden state
        x = x.flatten(1)  # flatten X to feed into Linear layer
        x = self.fc(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            x, _ = self.rnn(x, self.h) # Run thru RNN
            x = x.flatten(1)  # flatten X to feed into Linear layer
            x = self.fc(x)
            x = F.softmax(x, dim=1) # Convert to probabilities
            return x
    
    def loss(self, probs, labels):
        # Define the loss function
        loss = nn.CrossEntropyLoss(reduction=self.loss_reduction)
        return loss(probs, labels)

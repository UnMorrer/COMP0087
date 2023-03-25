# LSTM model for classification, based on BERT embeddings
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnected, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Define the forward pass of the neural network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def loss_fn(self, outputs, targets):
        # Define the loss function
        loss = nn.BCEWithLogitsLoss()
        return loss(outputs, targets)

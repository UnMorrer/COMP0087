# Fully connected model for classification, based on BERT embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
            self.hidden = torch.zeros(self.bidirectional_multiplier*rnn_layers,
                                 batch_size,
                                 hidden_size).to(device)
        else:
            self.hidden = init_hidden_state.to(device)

        # Linear/Dense/Fully Connected layer for RNN output
        self.fc = nn.Linear(
            # shape [batch_size, self.bidirectional_multiplier*hidden_size*tokens_per_essay]
            self.bidirectional_multiplier*hidden_size*max_tokens_per_essay,
            num_classes)
        
        # Other things to save
        self.loss_reduction = loss_reduction
        
    def forward(self, x):
        # Define the forward pass of the neural network
        x, self.hidden = self.rnn(x, self.hidden) # Run thru RNN + update hidden state
        x = x.flatten(1)  # flatten X to feed into Linear layer
        x = self.fc(x)

        # Detach the hidden state
        self.hidden = self.hidden.detach()
        return x
    
    def predict(self, x):
        with torch.no_grad():
            x, _ = self.rnn(x, self.hidden) # Run thru RNN
            x = x.flatten(1)  # flatten X to feed into Linear layer
            x = self.fc(x)
            x = F.softmax(x, dim=1) # Convert to probabilities
            return x
    
    def loss(self, probs, labels):
        # Define the loss function
        loss = nn.CrossEntropyLoss(reduction=self.loss_reduction)
        return loss(probs, labels)


# Use CNNs
# Base idea taken from: https://machinelearningmastery.com/best-practices-document-classification-deep-learning/
class CNNConnected(nn.Module):
    def __init__(self,
        input_size,
        cnn_kernel_size,
        cnn_stride,
        pooling_kernel_size,
        pooling_stride,
        num_classes,
        max_tokens_per_essay,
        padding_size=0,
        dilation=1,
        padding_mode="zeros",
        loss_reduction="mean",
    ):
        super(CNNConnected, self).__init__()

        # Helper functions
        def get_nth_number(x, n):
            # Helper function as stride etc. can be both 1 number or tuple/list
            if isinstance(x, (list, tuple)):
                return x[n]
            else:
                return x
            
        def calc_conv_out_size(input, padding, dilation, kernel_size, stride):
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            out = (input
            + 2 * padding
            - dilation
            * (kernel_size - 1)
            - 1
            ) / stride + 1

            return out

        # CNN to get features out
        self.cnn = nn.Conv2d(
            in_channels=1, 
            out_channels=1,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=padding_size,
            padding_mode=padding_mode,
            dilation=dilation)
        
        # Convolution input shape:
        # Width: max_tokens_per_essay
        # Height: input_size

        self.cnn_h_out = calc_conv_out_size(
            max_tokens_per_essay,
            get_nth_number(padding_size, 0),
            get_nth_number(dilation, 0),
            get_nth_number(cnn_kernel_size, 0),
            get_nth_number(cnn_stride, 0)
        )

        self.cnn_w_out = calc_conv_out_size(
            input_size,
            get_nth_number(padding_size, 1),
            get_nth_number(dilation, 1),
            get_nth_number(cnn_kernel_size, 1),
            get_nth_number(cnn_stride, 1)
        )

        # Roundings
        self.cnn_h_out = np.floor(self.cnn_h_out)
        self.cnn_w_out = np.floor(self.cnn_w_out)

        # Max pooling for convolution
        self.mp2d = nn.MaxPool2d(
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            padding=padding_size,
            dilation=dilation,
        )

        self.pooling_h_out = calc_conv_out_size(
            self.cnn_h_out,
            get_nth_number(padding_size, 0),
            get_nth_number(dilation, 0),
            get_nth_number(pooling_kernel_size, 0),
            get_nth_number(pooling_stride, 0)
        )

        self.pooling_w_out = calc_conv_out_size(
            self.cnn_w_out,
            get_nth_number(padding_size, 1),
            get_nth_number(dilation, 1),
            get_nth_number(pooling_kernel_size, 1),
            get_nth_number(pooling_stride, 1)
        )
        # Roundings
        self.pooling_h_out = np.floor(self.pooling_h_out).astype(int)
        self.pooling_w_out = np.floor(self.pooling_w_out).astype(int)

        # Linear/Fully connected layer
        self.fc = nn.Linear(
            # shape [batch_size, self.bidirectional_multiplier*hidden_size*tokens_per_essay]
            self.pooling_h_out * self.pooling_w_out,
            num_classes)
        
        # Loss
        self.loss_reduction = loss_reduction

    def forward(self, x):
        x = x.unsqueeze(1)  #Add signal channel
        x = self.cnn(x) # Apply CNN
        x = self.mp2d(x) # Max pooling of CNN output
        x = F.relu(x) # RelU activation
        x = x.flatten(1) # Flatten before passing into linear layer
        x = self.fc(x) # Fully connected -> probabilities

        return(x)

    def predict(self, x):
        x = self.cnn(x) # Apply CNN
        x = self.mp2d(x) # Max pooling of CNN output
        x = F.relu(x) # RelU activation
        x = self.fc(x) # Fully connected -> probabilities

        return(x)

    def loss(self, probs, labels):
        # Define the loss function
        loss = nn.CrossEntropyLoss(reduction=self.loss_reduction)
        return loss(probs, labels)

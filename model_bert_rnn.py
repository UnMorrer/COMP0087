# Orchestration for BERT LSTM model

# Base packages
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Custom packages
import src.models.bert as bert
import src.models.model_trainer as model_utils

# Settings
input_size = 768 # size of the BERT-encoded input
hidden_size = 128
num_classes = 2
num_epochs = 50
max_tokens = 512
tokenizer_model_name = "bert-base-uncased"
batch_size = 64
epochs = 50
lr = 0.1

# Model-related things
model = bert.RNNConnected(
    input_size,
    hidden_size,
    num_classes,
    batch_size,
    max_tokens
    )
optimizer = torch.optim.Adam

# Tokenization
tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
tokenizer_model = BertModel.from_pretrained(tokenizer_model_name)

# Call model trainer function
model_trainer(
    torch_model_object=model,
    batch_size=batch_size,
    epochs=num_epochs,
    tokenizer_object=tokenizer,
    tokenizer_model_object=tokenizer_model,
    optimizer_object=optimizer,
    learning_rate=lr,
    max_essay_tokens=max_tokens,
    model_save_name="example",
    training_device='cpu',
    padding_strategy="right",
    truncation_strategy="end",
    checkpoints_enabled=False,
    model_save_dir="models"
)
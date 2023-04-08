# Orchestration for BERT RNN model

# Base packages
import torch
from transformers import BertTokenizer, BertModel

# Custom packages
import src.models.bert as bert
import src.models.utils as model_utils

# Settings
input_size = 768 # size of the BERT-encoded input
hidden_size = 128
num_classes = 2
num_epochs = 100
max_tokens = 256
tokenizer_model_name = "bert-base-uncased"
batch_size = 64
lr = 0.00001

# Model-related things
model = bert.RNNConnected(
    input_size,
    hidden_size,
    num_classes,
    batch_size,
    max_tokens,
    device="cuda"
    )
optimizer = torch.optim.Adam

# Tokenization
tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
tokenizer_model = BertModel.from_pretrained(tokenizer_model_name)

# Call model trainer function
model_utils.model_trainer(
    torch_model_object=model,
    batch_size=batch_size,
    epochs=num_epochs,
    tokenizer_object=tokenizer,
    tokenizer_model_object=tokenizer_model,
    optimizer_object=optimizer,
    learning_rate=lr,
    max_essay_tokens=max_tokens,
    model_save_name="bert_rnn_256",
    training_device="cuda",
    padding_strategy="right",
    truncation_strategy="end",
    checkpoints_enabled=False,
    model_save_dir="models"
)
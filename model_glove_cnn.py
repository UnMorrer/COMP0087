# Orchestration for BERT RNN model

# Base packages
import torch 
from transformers import BertTokenizer, BertModel

# Custom packages
import src.models.bert as bert
import src.models.utils_glove as model_utils

# Settings
# input_size = 768 # size of the BERT-encoded input
input_size = 300 # size of the BERT-encoded input
# cnn_kernel_size = (64, 768) # Token kernel, Embedding input kernel
cnn_kernel_size = (64, 300) # Token kernel, Embedding input kernel
cnn_stride = (16, 1) # , token stride, embedding stride
pool_kernel_size = (7, 1)
pool_stride = (3, 1)
num_classes = 2
num_epochs = 100
max_tokens = 64
tokenizer_model_name = "bert-base-uncased"
batch_size = 64
lr = 0.001

# Model-related things
model = bert.CNNConnected(
    input_size,
    cnn_kernel_size,
    cnn_stride,
    pool_kernel_size,
    pool_stride,
    num_classes,
    max_tokens
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
    tokenizer_object=tokenizer, # change this
    tokenizer_model_object=tokenizer_model, # change this
    optimizer_object=optimizer,
    learning_rate=lr,
    max_essay_tokens=max_tokens,
    model_save_name="glove_cnn_64", # change this
    #training_device="cuda",
    training_device="cpu",# change this
    padding_strategy="right",
    truncation_strategy="end",
    checkpoints_enabled=False,
    model_save_dir="models"
)
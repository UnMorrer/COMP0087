# Base packages
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Custom code
import src.load.dataset_hf as load_data
import src.tokenization.general_hf_tokenizer as token_utils

# Variables
model_name = "roberta-base" # "roberta-large"
data = load_data.read_in(
    sample=True,
    sample_size=500
    )["validation"]
max_tokens = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prediction pipeline
model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = RobertaTokenizer.from_pretrained(
    model_name,
    padding='max_length',
    truncation=True,
    max_length=max_tokens
)

# Wrapping for tokenization
token_func = lambda x: token_utils.tokenizer_function(x, tokenizer=tokenizer, max_length=max_tokens)
tokenized_data = data.map(token_func, batched=True)

tokens = tokenized_data["input_ids"]
mask = tokenized_data["attention_mask"]

# Convert to PyTorch tensors
tokens = torch.tensor(tokens) # sample size x max_tokens tensors
mask = torch.tensor(mask)

# Model inference
with torch.no_grad():
    logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
    probs = logits.softmax(dim=-1) # Applies softmax, returns sample size x max_tokens tensor

# Get probabilities
fake = probs.detach().cpu()[:, 0]
real = probs.detach().cpu()[:, 1]

# Get label & compare

a = 1
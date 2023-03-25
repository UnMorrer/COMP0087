# Base packages
import torch
import numpy as np
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Custom code
import src.load.dataset_hf as load_data
import src.tokenization.general_hf_tokenizer as token_utils
import src.evaluation.utils as eval_utils

# Variables
model_name = "roberta-base" # "roberta-large"
data = load_data.read_in(
    sample=False
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

# Iterating over batches of answer for model inference
dataloader = torch.utils.data.DataLoader(
    data.select_columns(["answer", "generated", "index"]),
    batch_size=100,
    pin_memory=True)

# Keeping track of correct preds
probabilities = []
labels = []
correct_preds = 0
i = 0

for batch in dataloader:
    # Print batch information
    i += 1
    print(f"Working on batch num: {i}")

    # Tokenization
    tokenized_data = token_utils.tokenizer_function(
        batch["answer"],
        tokenizer=tokenizer,
        max_length=max_tokens)

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
    fake = probs.detach().cpu().numpy()[:, 0]
    ground_truth = np.array(batch["generated"])

    # Compare & get number of correct predictions
    correct = eval_utils.num_correct_predictions(fake, ground_truth)

    # Save correct preds and probs
    labels += np.array(batch["generated"]).tolist()
    correct_preds += correct
    probabilities += fake.tolist()

# Post - session analysis
df = pd.DataFrame(
        {
            "probs": probabilities,
            "labels": labels,
        }
    )

# TODO: Graph
a = 1
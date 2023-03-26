# Orchestration for BERT LSTM model

# Base packages
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Custom packages
import src.models.bert as bert
import src.load.dataset_hf as load_data
import src.tokenization.general_hf_tokenizer as token_utils
import src.evaluation.utils as eval_utils

# Settings
input_size = 768 # size of the BERT-encoded input
hidden_size = 128
num_classes = 2
num_epochs = 50
max_tokens = 512
tokenizer_model_name = "bert-base-uncased"
batch_size = 16
epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loader
data = load_data.read_in(
    sample=False
    )

train_dataloader = torch.utils.data.DataLoader(
    data["train"].select_columns(["answer", "generated"]),
    batch_size=batch_size,
    pin_memory=True
)

eval_dataloader = torch.utils.data.DataLoader(
    data["validation"].select_columns(["answer", "generated"]),
    batch_size=batch_size,
    pin_memory=True
)

# Model-related things
model = bert.RNNConnected(
    input_size,
    hidden_size,
    num_classes,
    batch_size,
    max_tokens
    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Tokenization
tokenizer = BertTokenizer.from_pretrained(
    tokenizer_model_name,
    padding='max_length',
    truncation=True,
    max_length=max_tokens
)

tokenizer_model = BertModel.from_pretrained(tokenizer_model_name).to(device)

for epoch in range(num_epochs):
    # Training loop
    for batch in train_dataloader:
        # Reset gradients
        optimizer.zero_grad()

        # Tokenize input
        tokenized_batch = token_utils.tokenizer_function(
            batch["answer"],
            tokenizer=tokenizer,
            max_length=max_tokens)
        input_vectors = token_utils.get_vector_representation(
            tokenized_batch["input_ids"],
            tokenizer_model
        ).to(device)
        # Shape of input_vectors:
        # <batch_size> x <num_tokens> x <encoding_size>
        outputs = model(input_vectors)

        # Convert classes to labels
        labels = batch["generated"].long() # 1 is Generated/Fake, 0 is Real
        labels = labels.to(device)
        loss = model.loss(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    correct = 0
    total = 0
    for batch in eval_dataloader:
        # Tokenize input
        tokenized_batch = token_utils.tokenizer_function(
            batch["answer"],
            tokenizer=tokenizer,
            max_length=max_tokens)
        
        input_vectors = token_utils.get_vector_representation(
            tokenized_batch["input_ids"],
            tokenizer_model
        ).to(device)

        outputs = model.predict(input_vectors)
        # Gather correct predictions
        fake = outputs.detach().cpu().numpy()[:, 1]
        ground_truth = np.array(batch["generated"])

        # Save predictions
        correct += eval_utils.num_correct_predictions(fake, ground_truth)
        total += len(ground_truth)

    accuracy = correct / total
    print(f'Epoch {epoch}, validation accuracy: {accuracy:.2f}')
    a = 1
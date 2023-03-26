# Orchestration for BERT LSTM model

# Base packages
import torch
from transformers import BertTokenizer, BertModel

# Custom packages
import src.models.bert as bert
import src.load.dataset_hf as load_data
import src.tokenization.general_hf_tokenizer as token_utils

# Settings
input_size = 768 # size of the BERT-encoded input
hidden_size = 256
num_classes = 2
num_epochs = 50
max_tokens = 512
tokenizer_model_name = "bert-base-uncased"
train_batch_size = 16
eval_batch_size = 64
epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loader
data = load_data.read_in(
    sample=False
    )

train_dataloader = torch.utils.data.DataLoader(
    data["train"].select_columns(["answer", "generated"]),
    batch_size=eval_batch_size,
    pin_memory=True
)

eval_dataloader = torch.utils.data.DataLoader(
    data["validation"].select_columns(["answer", "generated"]),
    batch_size=eval_batch_size,
    pin_memory=True
)

# Model-related things
model = bert.FullyConnected(input_size, hidden_size, num_classes).to(device)
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
        probs = model(input_vectors)
        loss = model.loss(probs, batch["generated"])
        loss.backward()
        optimizer.step()

    # Validation loop
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in eval_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch}, validation accuracy: {accuracy:.2f}')

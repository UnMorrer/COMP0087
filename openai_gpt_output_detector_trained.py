# NOTE: ABANDONED - Model doesn't expose/return loss - hence cannot be fine-tuned easily later down the line

# Base packages
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments

# Custom code
import src.load.dataset_hf as load_data
import src.tokenization.general_hf_tokenizer as token_utils
import src.evaluation.utils as eval_utils

# Variables
model_name = "roberta-base" # "roberta-large"
data = load_data.read_in(
    sample=False
    )
max_tokens = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plot_bins = 25
save_histogram = True
histogram_filepath = "visualizations/openai_trained_prob_distributions.png"
train_batch_size = 16
eval_batch_size = 64
epochs = 50

# Prediction pipeline
model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = RobertaTokenizer.from_pretrained(
    model_name,
    padding='max_length',
    truncation=True,
    max_length=max_tokens
)

# Tokenization
tokenizer_function = lambda x: token_utils.tokenizer_function(x["answer"],tokenizer=tokenizer,max_length=max_tokens)

train_data = data["train"]
tokenized_train = train_data.map(tokenizer_function, batched=True, batch_size=train_batch_size)

eval_data = data["validation"]
tokenized_eval = eval_data.map(tokenizer_function, batched=True, batch_size=eval_batch_size)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=epochs,              # total number of training epochs
    warmup_steps=10,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    evaluation_strategy='epoch',     # evaluation strategy to adopt during training
    gradient_checkpointing=True,      # to save GPU memory
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
)

# Define the trainer
trainer = Trainer(
    model=model,                      # the instantiated Transformers model to be trained
    args=training_args,               # training arguments, defined above
    train_dataset=tokenized_train,         # training dataset
    eval_dataset=tokenized_eval     # evaluation dataset
)

# Train the model
trainer.train()

# Dataloader for final evaluation
eval_dataloader = torch.utils.data.DataLoader(
    data["validation"].select_columns(["answer", "generated"]),
    batch_size=eval_batch_size,
    pin_memory=True
)

# Keeping track of correct preds
probabilities = []
labels = []
correct_preds = 0
i = 0

for batch in eval_dataloader:
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
            "generated": labels,
        }
    )

# Plotting data on histogram

# define data
probs_gen = df[df["generated"] == True]["probs"]
probs_real = df[df["generated"] == False]["probs"]

# initialize figure with subplots
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

# create first histogram and mean line
n0, bins0, patches0 = ax0.hist(
    probs_gen,
    bins=20,
    density=True,
    histtype='bar',
    color="tab:blue")
mean0 = np.mean(probs_gen)
ax0.axvline(x=mean0, color='r', linestyle='--', linewidth=2)
ax0.text(
    mean0,
    1.1 * np.max(n0),
    f"Mean: {mean0:.4f}",
    horizontalalignment='center')

# create second histogram and mean line
n1, bins1, patches1 = ax1.hist(
    probs_real,
    bins=20,
    density=True,
    histtype='bar',
    color="tab:orange")
mean1 = np.mean(probs_real)
ax1.axvline(x=mean1, color='r', linestyle='--', linewidth=2)
ax1.text(
    mean1,
    1.1 * np.max(n1),
    f"Mean: {mean1:.4f}",
    horizontalalignment='center')

# set titles and labels
ax0.set_title("Generated essays")
ax1.set_title("Real essays")
ax0.set_xlabel("Fake probability")
ax1.set_xlabel("Fake probability")
ax0.set_ylabel("Density")

# set common X and Y limits
xmin = min(np.min(bins0), np.min(bins1))
xmax = max(np.max(bins0), np.max(bins1))
ymin = 0
ymax = max(np.max(n0), np.max(n1))
ax0.set_xlim([xmin, xmax])
ax0.set_ylim([ymin, ymax])

# display plot
fig.tight_layout()
plt.show()

if save_histogram:
    plt.savefig(histogram_filepath)
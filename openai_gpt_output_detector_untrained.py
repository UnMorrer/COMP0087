# Base packages
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
plot_bins = 25
save_histogram = True
histogram_filepath = "visualizations/openai_base_prob_distributions.png"

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
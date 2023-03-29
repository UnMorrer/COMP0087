# Evaluation for models trained with BERT encodings

# General packages

# Custom packages
import src.load.dataset_hf as load

# Settings
file_loc = {
    "test": "data/essays_test.csv",
    "raw": "data/essays_raw.csv"
}

raw, test = load.read_in(
    sample=False,
    data_files=file_loc
)
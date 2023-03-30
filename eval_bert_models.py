# Evaluation for models trained with BERT encodings

# General packages
import torch
import os
from transformers import BertTokenizer, BertModel

# Custom packages
import src.load.dataset_hf as load
import src.evaluation.utils as eval
import src.models.bert as bert_models

# Settings
file_loc = {
    "test": "data/essays_test.csv",
    "raw": "data/essays_raw.csv"
}

eval_models = [
    "bert_rnn",
    "bert_cnn",
    "bert_lstm"
]

model_info = {
    "bert_rnn": {
        "torch_model_obj": bert_models.RNNConnected,
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_rnn_best"),
        "tokenizer_obj": BertTokenizer,
        "tokenizer_model_obj": BertModel,
    },
    "bert_lstm": {
        "torch_model_obj": bert_models.LSTMConnected,
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_lstm_best"),
        "tokenizer_obj": BertTokenizer,
        "tokenizer_model_obj": BertModel,
    },
    "bert_cnn": {
        "torch_model_obj": bert_models.CNNConnected,
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_cnn_best"),
        "tokenizer_obj": BertTokenizer,
        "tokenizer_model_obj": BertModel,
    }
}


# Load in data
data = load.read_in(
    sample=False,
    data_files=file_loc
)

raw = data["raw"]
test = data["test"]

# Bit of data engineering to keep only chatGPT output from raw set
chatGPT_test = raw.filter(lambda x: x["model"] == "chatGPT")

# NOTE: Setting for evaluation datasets
eval_datasets = [chatGPT_test, test]


# Evaluate all possible model X dataset combinations
for dataset in eval_datasets:
    for model_name in eval_models:

        print(f"Model: {model_name}")
        eval.model_tester(
                torch_model_obj=model_info[model_name]["torch_model_obj"],
                weights_file_path=model_info[model_name]["weights_file_path"],
                data=dataset,
                batch_size=64,
                tokenizer_obj=model_info[model_name]["tokenizer_obj"],
                tokenizer_model_obj=model_info[model_name]["tokenizer_model_obj"],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                max_essay_tokens=512,
                padding_strategy="right",
                truncation_strategy="end",
                print_scores=True
        )

a = 1
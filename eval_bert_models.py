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
    "bert_rnn_512",
    "bert_rnn_128",
    "bert_rnn_64",
    "bert_cnn_512",
    "bert_lstm_512",
    "bert_lstm_128",
    "bert_lstm_64"
]

tokenizer_model_name = "bert-base-uncased"

model_info = {
    "bert_rnn_512": {
        "torch_model_obj": bert_models.RNNConnected(
            input_size=768,
            hidden_size=128,
            num_classes=2,
            batch_size=64,
            max_tokens_per_essay=512,
            ),
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_rnn_best"),
        "tokenizer_obj": BertTokenizer.from_pretrained(tokenizer_model_name),
        "tokenizer_model_obj": BertModel.from_pretrained(tokenizer_model_name),
    },
    "bert_rnn_128": {
        "torch_model_obj": bert_models.RNNConnected(
            input_size=768,
            hidden_size=128,
            num_classes=2,
            batch_size=64,
            max_tokens_per_essay=128,
            ),
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_rnn_128_best"),
        "tokenizer_obj": BertTokenizer.from_pretrained(tokenizer_model_name),
        "tokenizer_model_obj": BertModel.from_pretrained(tokenizer_model_name),
    },
    "bert_rnn_64": {
        "torch_model_obj": bert_models.RNNConnected(
            input_size=768,
            hidden_size=128,
            num_classes=2,
            batch_size=64,
            max_tokens_per_essay=64,
            ),
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_rnn_64_best"),
        "tokenizer_obj": BertTokenizer.from_pretrained(tokenizer_model_name),
        "tokenizer_model_obj": BertModel.from_pretrained(tokenizer_model_name),
    },
    "bert_lstm_512": {
        "torch_model_obj": bert_models.LSTMConnected(
            input_size=768,
            hidden_size=128,
            num_classes=2,
            batch_size=64,
            max_tokens_per_essay=512,
            ),
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_lstm_best"),
        "tokenizer_obj": BertTokenizer.from_pretrained(tokenizer_model_name),
        "tokenizer_model_obj": BertModel.from_pretrained(tokenizer_model_name),
    },
    "bert_lstm_128": {
        "torch_model_obj": bert_models.LSTMConnected(
            input_size=768,
            hidden_size=128,
            num_classes=2,
            batch_size=64,
            max_tokens_per_essay=128,
            ),
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_lstm_128_best"),
        "tokenizer_obj": BertTokenizer.from_pretrained(tokenizer_model_name),
        "tokenizer_model_obj": BertModel.from_pretrained(tokenizer_model_name),
    },
    "bert_lstm_64": {
        "torch_model_obj": bert_models.LSTMConnected(
            input_size=768,
            hidden_size=128,
            num_classes=2,
            batch_size=64,
            max_tokens_per_essay=64,
            ),
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_lstm_64_best"),
        "tokenizer_obj": BertTokenizer.from_pretrained(tokenizer_model_name),
        "tokenizer_model_obj": BertModel.from_pretrained(tokenizer_model_name),
    },
    "bert_cnn_512": {
        "torch_model_obj": bert_models.CNNConnected(
            input_size=768,
            cnn_kernel_size=(64, 768),
            cnn_stride=(16, 1),
            pooling_kernel_size=(7, 1),
            pooling_stride=(3, 1),
            num_classes=2,
            max_tokens_per_essay=512,
        ),
        "weights_file_path": os.path.join(os.getcwd(), "models", "bert_cnn_best"),
        "tokenizer_obj": BertTokenizer.from_pretrained(tokenizer_model_name),
        "tokenizer_model_obj": BertModel.from_pretrained(tokenizer_model_name),
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
eval_datasets = [test] #, chatGPT_test]


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
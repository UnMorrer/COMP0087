# Based on: https://github.com/openai/gpt-2-output-dataset

# General packages
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch_inference(
        text,
        model,
        tokenizer,
        max_tokens,
        device
):
    """
    Wrapper for collating inference output from OpenAI
    GPT-2 detector

    Inputs:
    text - [str]: List of essays to analyze
    model - hf.Model: GPT2- detector model HuggingFace
    tokenizer - hf.Tokenizer: RoBERTa/Huggingface tokenizer
    max_tokens - int: Maximum number of tokens for input

    Outputs:
    preds - torch.tensor: A tensor with size 2xlength(text)
    that returns the fake and real probabilities generated
    by the model (in this order).
    """
    tokens = tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_tokens)

    token_ids = tokens.data["input_ids"]
    mask = tokens.data["attention_mask"]

    with torch.no_grad():
        logits = model(token_ids.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    return probs # fake, real

if __name__ == "__main__":
model_name = "roberta-base"
get_batch_inference(
    ["This is an example sentence.", "This is yet another example sentence."],
    RobertaForSequenceClassification.from_pretrained(model_name).to(device),
    RobertaTokenizer.from_pretrained(model_name),
    10,
    device
)
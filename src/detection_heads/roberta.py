from transformers import RobertaForSequenceClassification, RobertaTokenizer

model_name = "roberta-base"
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
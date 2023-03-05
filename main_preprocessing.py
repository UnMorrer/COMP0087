# Orchestration for pre-processing input (to tokenizer)

# Example configuration
# ner = trfs.pipeline("ner", grouped_entities=True) # TODO: Specify model revision + framework
# resp = ner("This is a test text to see how much this encoding can handle. I started working for Dr. Hammerschmidt in February 2023 for a salary of £60,000 ($75,000). My e-mail address is hammerschmidt@gmail.com")

import src.tokenization.preprocessing as preproc
import src.load.train_test_split as tts

train, test, validate = tts.create_test_train_split()

ai_df = train[train["model"] == "text-davinci-003"]
ai_texts = ai_df["answer"].tolist()

preprocessed_texts = []

for text in ai_texts:
    preprocessed_text = preproc.ai_text_preprocessing(text)
    preprocessed_texts.append(preprocessed_text)

a = 1
# Orchestration for pre-processing input (to tokenizer)

# Example configuration
# ner = trfs.pipeline("ner", grouped_entities=True) # TODO: Specify model revision + framework
# resp = ner("This is a test text to see how much this encoding can handle. I started working for Dr. Hammerschmidt in February 2023 for a salary of £60,000 ($75,000). My e-mail address is hammerschmidt@gmail.com")

# Common libraries
import transformers
import os

# Customer packages
import src.tokenization.preprocessing as preproc
import src.load.train_test_split as tts

# Settings
model_name = "dslim/bert-base-NER"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)

# Options
entity_conversions = {
    "B-PER": "@PER",
    "I-PER": "",
    "B-LOC": "@LOC",
    "I-LOC": "",
    "B-ORG": "@ORG",
    "I-ORG": "",
    "B-MISC": "@MISC",
    "I-MISC": ""
}
save_folder = os.getcwd() + "/data/"

###########
# Actual code
###########

train, test, validation = tts.create_test_train_split()

data = [train, test, validation]
names = ["train", "test", "validation"]

for df, name in zip(data, names):
    answers = df["answer"]
    preproc_answers = []
    generated = df["generated"]
    
    # Status printout
    total_rows = len(df)
    i = 0

    for answer, generated_indicator in zip(answers, generated):
        # Select appropriate preprocessor
        if generated_indicator:
            preproc_answer = preproc.ai_text_preprocessor(
                answer,
                model=model,
                tokenizer=tokenizer,
                conversion=entity_conversions)
        else:
            preproc_answer = preproc.human_text_preprocessor(answer)

        # Save answer
        preproc_answers.append(preproc_answer)

        i += 1
        if i % 10 == 0:
            print(f"Preprocessing {name} set: {i}/{total_rows}")

    # Update answers
    df["answer"] = preproc_answers

    # Save df
    filename = save_folder + name + ".csv"
    df.to_csv()
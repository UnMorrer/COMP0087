# Named Entity Recognition for pipeline -> align test/train dataset
# Walk-through: https://huggingface.co/course/chapter7/2?fw=pt

# Explore how many unique tokens we have
import os
import pandas as pd
import re

def merged_data():
    """
    OS-agnostic function to load merged data
    """
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "data", "essays.csv")

    df = pd.read_csv(file_path, index_col=0)

    return df

df = merged_data()

human_answers = df[df["generated"] == False]["answer"].tolist()

# What to look for
# @[A-Z][0-9] -> @ followed by ALL CAPS (no numbers - no need to be unique)
regex = re.compile(r"@[A-Z]+")
masks = set()

for text in human_answers:
    mask_list = regex.findall(text)
    for mask in mask_list:
        masks.add(mask)

# Unique labels in human text: print(masks)
# '@MONTH', '@CAPS', '@MONEY', '@PERSON', '@PERCENT', '@STATE', '@LOCATION', '@NUM', '@DATE', '@EMAIL', '@CITY', '@DR', '@ORGANIZATION', '@TIME'
# @DR -> Dr. @DR (so prolly person name)

gpt_answers = df[df["model"] == "text-davinci-003"]["answer"].tolist()

# What to look for
# [...] -> Input enclosing
regex = re.compile(r"\[.*\]", re.DOTALL)
masks = set()

for text in gpt_answers:
    mask_list = regex.findall(text)
    for mask in mask_list:
        masks.add(mask)

# Unique labels in chatGPT text: print(masks)
# [Name of Newspaper], [Your Name], [Name], [NAME], [name], [YOUR NAME], [your name], [Insert Newspaper Name], [a book], [to remove a book], [Editor]
# + Gibbidy goo that is worth investigating
# [one], [with], [the book that we abhor], [a]

# ChatGPT prompts/tokens (such as [Your Name])
chatgpt_answers = df[df["model"] == "chatGPT"]["answer"].tolist()

# What to look for
# [...] -> Input enclosing
regex = re.compile(r"\[.*\]", re.DOTALL)
masks = set()

for text in chatgpt_answers:
    mask_list = regex.findall(text)
    for mask in mask_list:
        masks.add(mask)

# Unique labels in chatGPT text: print(masks)
# [Your Name]
# BUT, also names of IRL people are present!
a = 1

# Max sentence length
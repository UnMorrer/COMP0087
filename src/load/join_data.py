import pandas as pd
import os
import re

# Simple function to join Thibaud, Boris + Marcell data into a single .csv
thibaud_data_path = "/data/data_thibaud/"
boris_data_path = "/data/data_boris/"
marcell_data_path = "/data/responses/"
essay_train_data_path = "/data/asap-aes/training_set_rel3.xlsx"
essay_valid_data_path = "/data/asap-aes/valid_set.xlsx"
save_filepath = "/data/essays.csv"
questions = ["q1", "q2", "q7", "q8"]

# Thibaud + Boris -> CSV by q1, q2, q7, q8 texts
# Marcell -> Individual .txt files by question

# .csv column structure
out_cols = [
    "answer",
    "model",
    "temperature",
    "question",
    "generated"
]

human_cols = ["essay", "essay_set"]
human_cols_rename = ["answer", "question"]

cwd = os.getcwd()
out_df = pd.DataFrame(columns=out_cols)

def load_chatGPT(
        folder_path,
        regex):
    """
    Simple function to load individual .txt files into DataFrame

    inputs:
    folder_path - str: Relative path to folder where .txt files are located
    regex - str: String that will be converted to regular expression.
    All items matching the regex will be loaded & returned

    Returns:
    essays - list[str]: A list with all associated essays. 
    Each list item is an individual essay
    """

    file_list = os.listdir(folder_path)
    matching_files = list(filter(regex.match, file_list))
    essays = []

    for file in matching_files:
        with open(folder_path + file, "r") as file:
            essays.append(file.readlines())
    
    return essays

for question in questions:
    # Load Thibaud/Boris model
    df_thibaud = pd.read_excel(cwd + thibaud_data_path + question + ".xlsx")
    df_boris = pd.read_excel(cwd + boris_data_path + question + ".xlsx")

    # Bit of selection required to select correct essay set
    df_train = pd.read_excel(cwd + essay_train_data_path)
    df_train = df_train[df_train["essay_set"] == int(question[-1])]
    df_valid = pd.read_excel(cwd + essay_valid_data_path)
    df_valid = df_valid[df_valid["essay_set"] == int(question[-1])]

    # Load chatGPT outputs
    list_chatgpt = load_chatGPT(
        cwd + marcell_data_path,
        re.compile(rf"{question}.*\.txt$"))

    # Some renaming before joining etc.
    df_gpt3 = pd.concat([df_thibaud, df_boris])
    df_gpt3.columns = out_cols[0:-1]
    df_gpt3[out_cols[-1]] = True

    # For human essays
    df_train = df_train[human_cols]
    df_valid = df_valid[human_cols]

    df_human = pd.concat([df_train, df_valid])
    df_human.columns = human_cols_rename
    df_human[out_cols[-1]] = False

    df_chatgpt = pd.DataFrame(data={
        "answer": list_chatgpt,
    })
    df_chatgpt["model"] = "chatGPT"
    df_chatgpt["question"] = int(question[-1])
    df_chatgpt[out_cols[-1]] = True

    # Concatenate results into single df
    out_df = pd.concat([out_df, df_gpt3, df_chatgpt, df_human])

# Save results
out_df.to_csv(cwd + save_filepath)
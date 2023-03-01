import pandas as pd
import os

# Simple function to join Thibaud, Boris + Marcell data into a single .csv
thibaud_data_path = "/data/data_thibaud/"
boris_data_path = "/data/data_boris/"
marcell_data_path = "/data/responses/"
essay_train_data_path = "/data/asap-aes/training_set_rel3.xlsx"
essay_valid_data_path = "/data/asap-aes/valid_set.xlsx"

questions = ["q1", "q2", "q7", "q8"]

# Thibaud + Boris -> CSV by q1, q2, q7, q8 texts
# Marcell -> Individual .txt files by question

# .csv column structure
out_cols = [
    "answer",
    "model",
    "temperature",
    "question",
    "fake"
]

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
        with open(file, "r") as file:
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
    list_chatgpt = load_chatGPT(cwd + marcell_data_path, f"{question}*.txt$")

# Should also join student essay data into same .csv file
# Load train, test and validation dataset using pandas dataframe
import src.load.read_data as read
import pandas as pd
import numpy as np

def create_test_train_split(
    questions = [1, 2, 7, 8],
    generation_models = ["text-davinci-003"],
    random_seed = 42,
    replacement = False,
    n = 900,
    train_frac = 0.7,
    validation_frac = 0.2
):
    """
    Function to create balanced sample + test/train/validation splitm from data

    Inputs:
    questions - list(int): Question number. Which essay questions to include in the sample?
    generation_models - list(str): Allowed generation models to sample from.
    random_seed - int: Seed for the random sampling
    replacement - bool: Is sampling done with replacement?
    n - int: Number of samples for each question x ai/human cross-section
    train_frac - float: Proportion of training samples (train num: n x train_frac)
    validation_frac - float: Proportion of validation samples 

    Returns:
    train_df, test_df, validation_df
    pd.DataFrame containing a balanced sample (size n) for each model & question sampled.

    """
    # Train, test, (validation) proportions
    test_frac = 1 - train_frac - validation_frac # What is left

    df = read.merged_data()
    sample_df = pd.DataFrame()

    for question_num in questions:
            ai_sample_df = df[(df["question"] == question_num) & (df["model"].isin(generation_models))]
            human_sample_df = df[(df["question"] == question_num) & (df["model"].isna())]

            ai_full_sample = ai_sample_df.sample(n=n, replace=replacement, random_state=random_seed)
            human_full_sample = human_sample_df.sample(n=n, replace=replacement, random_state=random_seed)

            sample_df = pd.concat([ai_full_sample, human_full_sample, sample_df])
    
    # Create train/test split
    train, validate, test = np.split(
        sample_df.sample(frac=1, random_state=random_seed),
        [int(train_frac*len(sample_df)), int((train_frac + test_frac)*len(sample_df))]
        )

    # Return results
    return train, validate, test
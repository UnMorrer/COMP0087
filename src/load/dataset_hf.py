from datasets import load_dataset

# Default settings
data_files = {
    "train": "data/essays_train.csv",
    "test": "data/essays_test.csv",
    "validation": "data/essays_validation.csv"}

def read_in(
    sample : bool,
    sample_size=1000,
    data_files=data_files,
    streaming=False,
    random_seed=42
):
    """
    Simple function to read in our dataset using
    Hugging Face's datasets library

    Inputs:
    sample - bool: Whether to take (random)
    sample of the data
    sample_size - int: Size of sample (if sampled)
    data_files - dict(str): List of data files
    and their relative name/location as stored on disk
    streaming - bool: When set to True, returns an 
    iterable that can be used to get <batch_size> long
    samples from the full dataset.
    random_seed - int: Random seed for sampling

    Returns:
    data - DataSetDict or IterableDatasetDict with data loaded.
    Docs are here: https://huggingface.co/docs/datasets/v2.10.0/en/package_reference/main_classes#datasets.DatasetDict
    https://huggingface.co/docs/datasets/v2.10.0/en/package_reference/main_classes#datasets.IterableDatasetDict
    """

    # Load dataset
    data = load_dataset("csv", data_files=data_files)

    # Rename "Unnamed: 0" to index
    data = data.rename_column(
        original_column_name="Unnamed: 0", new_column_name="index"
    )

    # Shuffle data
    data = data.shuffle(seed=random_seed)

    # Take random sample 
    if sample:
        for k in data.keys():
            data[k] = data[k].select(range(sample_size))
    
    return data

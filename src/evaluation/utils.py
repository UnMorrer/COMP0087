# Utilities for model evaluation

# General packages
import torch

# Custom packages
import src.tokenization.general_hf_tokenizer as token_utils


# Function to test models on test set & chatGPT output
def model_tester(
        torch_model_obj,
        weights_file_path,
        data,
        batch_size,
        tokenizer_obj,
        tokenizer_model_obj,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_essay_tokens=512,
        padding_strategy="right",
        truncation_strategy="end",
        print_scores=True
):
    """
    Function to evaluate model performance. Intended for chatGPT
    output evaluation and TEST set evaluation

    Inputs:
        torch_model_obj - class inheriting after nn.Module: PyTorch
    class of model implementation. MUST HAVE:
            - .forward() method for running data through the model
            - .predict() method for obtaining evaluation predictions,
            above should NOT update model weights
            - .loss() function that returns model loss for batch
            batch_size - int: Batch size for model training & evaluation.
        weights_file_path - str/path obj: Path for model weights file
        data - transformers DataSet: Dataset that contains data
    the model will be evaluated against. MUST contain "generated" and
    "answer" columns.
        batch_size - int: Evaluation batch size. Should be the same
    as the batch size used for model training.
    tokenizer_obj - AutoTokenizer from transformers library.
    This tokenizer should be pre-initialized with model name
        tokenizer_model_obj - AutoModel from transformers library.
    This model object should be initialized with model name. It must
    take input_ids from the tokenizer.
        device - str: Name of device the model will be run on. All data related
    to training (model, data etc.) will be moved to it.
        max_essay_tokens - int: The number of tokens (~words) that are
    processed for each essay. Essays containing fewer tokens than this
    will be padded according to "padding strategy". Essays containing
    more tokens than this will be truncated according to "truncation
    strategy".
        padding_strategy - str: padding direction. "right" pads the end
    of the sentence to reach max_length tokens. "left" pads the beginning
    of the sentence to reach max_tokens length.
        truncation_strategy - str: Truncation direction. "end" removes
    tokens from the end when sentences are too long. "beginning"
    removes tokens from the beginning when they are too long.
        print_scores - bool: Whether to print calculated scores
    (accuracy, recall, precision, f1) to console

    Returns:
    acc - float: Model accuracy
    recall - float: Model recall
    precision - float: Model precision
    F1 - float: Model F1 score

    """

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        data["validation"].select_columns(["answer", "generated"]),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Load model from saved state dictionary
    model = torch_model_obj.to(device)
    model.load_state_dict(torch.load(weights_file_path))
    model.eval()

    # Tokenization
    tokenizer_model = tokenizer_model_obj.to(device)

    # Gathering inference in batches
    matrix = torch.zeros(2,2)
    for batch in dataloader:
        # Tokenize input
        tokenized_batch = token_utils.tokenizer_function(
            batch["answer"],
            tokenizer=tokenizer_obj,
            max_length=max_essay_tokens,
            padding=padding_strategy,
            truncation=truncation_strategy)
        
        input_vectors = token_utils.get_vector_representation(
            tokenized_batch["input_ids"],
            tokenizer_model
        ).to(device)

        # Run input thru model
        outputs = model.predict(input_vectors)

        # Gather correct predictions
        pred = outputs.detach().cpu()
        pred = torch.clamp(pred.round(), 0, 1).squeeze(1).int()
        batch['label'] = batch['label'].to(device)
        for i in range(pred.shape[0]):
            matrix[pred[i]][batch['generated'][i].int()] += 1
    
    # Calculate evluation results
    acc = matrix.trace()/matrix.sum()
    recall = matrix[1,1]/matrix[1,:].sum()
    precision = matrix[1,1]/matrix[:,1].sum()
    F1 = 2 * precision*recall/(precision+recall)

    # Print evaluation results
    if print_scores:
        print('Accuracy: ', acc)
        print('Recall: ', recall)
        print('Precision: ', precision)
        print('F1: ', F1)
        print(matrix)

    # Return stuff
    return acc, recall, precision, F1

def num_correct_predictions(
        probs,
        labels,
        prob_type="fake",
        threshold=0.5):
    """
    Function evaluating prediction accuracy.

    Inputs:
    probs - np.array(float): 1x<batch_size> long NumPy array with 
    predicted probabilities for <prob_type>
    labels - [bool]: NumPy array of boolean values with ground truth.
    Fake/Generated essays should be 'True' and real/human-written
    essays should be 'False' values. Corresponds to "generated"
    column of dataset.
    prob_type - str: 'fake' or 'real' probability type for input
    threshold - float: Probability threshold to compare against.
    prob > threshold will be classified as <prob_type>
    prob < threshold will be classified as <other_prob_type>

    Returns:
    num_correct - int: Number of correct predictions for model
    """
    # Convert real probability to fake probability - if required
    if prob_type.lower() == "real":
        probabilities = 1 - probs
    elif prob_type.lower() == "fake":
        probabilities = probs
    else:
        raise ValueError(f"Invalid argument for prob_type: {prob_type}. Please use either 'real' or 'fake'.")
    
    # Compare fake probs with labels
    predictions = probabilities > threshold
    correct = predictions == labels

    # Convert to integer and sum up
    int_array = correct.astype(int)
    num_correct = sum(int_array)

    return num_correct

# Example usage
if __name__ == '__main__':
    pass
# Utilities for model evaluation

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
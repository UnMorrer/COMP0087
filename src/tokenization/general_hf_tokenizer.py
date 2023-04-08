# Tokenization implemented using hugging face tokenizers
def tokenize_input(text,
                   num_tokens,
                   tokenizer,
                   model,
                   tensor_framework="pt"):
    """
    Simple wrapper for BERT tokenization

    Inputs:
    text - [str]: List, with string inputs (texts) to tokenize
    num_tokens - int: Number of tokens per text. Vectors shorter
    than this will be padded, vectors longer than this will be truncated
    tokenizer - transfomers Bert/Auto/...Tokenizer: Used for generating
    tokens and input/word IDs for the model.
    model - transformers Bert/Auto/...Model: Used for retrieving the
    word embeddings (vectors) associated with each word ID
    tensor_framework - str: Which framework/ml library to return
    tensors in (pt -> PyTorch, tf -> TensorFlow)

    Returns:
    encoded_input - Tensor of the specified framework: It has dimensions
    [batch_size x num_tokens x model_encoding_size]
    """

    encoded_input = tokenizer(
        text,
        return_tensors=tensor_framework,
        padding='max_length',
        truncation=True,
        max_length=num_tokens)
    token_ids = encoded_input.data["input_ids"]

    embedding_matrix = model.embeddings.word_embeddings.weight
    vector_rep = embedding_matrix[token_ids]

    return vector_rep


def tokenizer_function(input_text,
                       tokenizer,
                       max_length=0,
                       padding="right",
                       truncation="end"):
    """
    Function to tokenize input. Returns function
    so that map() can be applied. Figures out batch size automatically.

    Inputs:
    input_text - [str]: List of string inputs
    to pass down to tokenizer
    tokenizer - hf.Tokenizer: Tokenizer for model
    max_length - int: Maximal number of tokens
    per sentence. Defaults to 0 (all tokens)
    padding - str: padding direction. "right" pads
    the end of the sentence to reach max_length tokens.
    "left" pads the beginning of the sentence to reach
    max_tokens length
    padding_token - int: Token ID for padding token
    truncation - str: Truncation direction. "end" removes
    tokens from the end when sentences are too long. "beginning"
    removes tokens from the beginning when they are too long

    Returns:
    tokens - [int]: List of integers, containing token IDs.
    """

    # Determine if padding/truncation is required
    if max_length == 0:
        return tokenizer(input_text)
    
    # Handle padding & truncation
    tokens = tokenizer(input_text)
    batch_size = len(tokens["input_ids"])
    padding_token=tokenizer.pad_token_id

    # Per-input based padding/truncation
    for i in range(batch_size):
        length = len(tokens["input_ids"][i])

        # Padding required
        if length < max_length:
            if padding.lower() == "right":
                tokens["input_ids"][i] = tokens["input_ids"][i] + (max_length - length) * [padding_token]
                tokens["attention_mask"][i] = tokens["attention_mask"][i] + (max_length - length) * [0]

            elif padding.lower() == "left":
                tokens["input_ids"][i] = (max_length - length) * [padding_token] + tokens["input_ids"][i]
                tokens["attention_mask"][i] = (max_length - length) * [0] + tokens["attention_mask"][i]

            else:
                raise ValueError(f"Invalid value for padding: {padding}. Please use 'left' or 'right' instead.")

        # Truncation required
        elif length > max_length:
            if truncation.lower() == "end":
                tokens["input_ids"][i] = tokens["input_ids"][i][:max_length]
                tokens["attention_mask"][i] = max_length * [1]

            elif truncation.lower() == "beginning":
                tokens["input_ids"][i] = tokens["input_ids"][i][length-max_length:]
                tokens["attention_mask"][i] = max_length * [1]
            
            else:
                raise ValueError(f"Invalid value for truncation: {truncation}. Please use 'end' or 'beginning' instead.")
        
    # Return result
    return tokens


def get_vector_representation(
        token_ids,
        model):
    """
    Function to get vector representation of/for set input IDs.
    Intended to compliment functionality of tokenizer_function
    (see above).

    Inputs:
    input_ids - [int]: List of input IDs that correspond uniquely
    to tokens, such as those returned by transformers tokenizers.
    model - transformers.model: An initialized model from the
    transformers library that corresponds to the tokenizer used.
    Weights can then be extracted from this model to use.

    Returns:
    vector_rep - torch.tensor: A <batch_size> x <model_hidden_rep>
    size tensor that contains the hidden vector representations
    for each token in all of the essays in the batch
    """
    embedding_matrix = model.embeddings.word_embeddings.weight
    vector_rep = embedding_matrix[token_ids, ]

    return vector_rep

if __name__ == "__main__":
    # Example usage
    from transformers import BertModel, BertTokenizer

    ex1 = "This is an example sentence"
    ex2 = "This is another example sentence with different length"

    text = [ex1, ex2]

    vector_rep = tokenize_input(
        text,
        350,
        BertTokenizer.from_pretrained('bert-base-uncased'),
        BertModel.from_pretrained("bert-base-uncased")
        )
    
    # Usage of tokenizer_function
    token_func = lambda x: tokenizer_function(x, tokenizer=BertTokenizer, max_length=16)
    tokenized_data = map(token_func, text)

    # Usage of get_vector_representation
    vector_rep = get_vector_representation(tokenized_data["input_ids"], BertModel.from_pretrained("bert-base-uncased"))
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
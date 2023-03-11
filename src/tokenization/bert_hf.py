# BERT Tokenization implemented using hugging face tokenizer
from transformers import BertTokenizer

def tokenize_input(text,
                   num_tokens,
                   tensor_framework="pt"):
    """
    Simple wrapper for BERT tokenization

    Inputs:
    Text - str: Input text to tokenize
    tensor_framework - str: Which framework/ml library to return
    tensors in (pt -> PyTorch, tf -> TensorFlow)

    Returns:
    encoded_input - :
    PyTorh tensors: [batch_size x num_tokens x 768]
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, return_tensors=tensor_framework)

    # TODO: Update code
    # TODO: Implement padding & truncation
    # TODO: Batch size - aware of batching

    return encoded_input

if __name__ == "__main__":
    from transformers import BertModel

    # Test playground
    essay = """
    Dear Editor,

    I am writing to express my opinion on the impact of computers on society. As we all know, technology has advanced rapidly in recent years, and computers have become an integral part of our daily lives. While some people argue that computers have a negative effect on society, I firmly believe that they have a positive impact.
    Computers have greatly improved our ability to learn and communicate. They provide us with access to a vast amount of information and allow us to connect with people from all over the world. With the help of computers, we can learn about different cultures, history, and geography. This has broadened our horizons and increased our understanding of the world.
    Moreover, computers have also helped to improve our hand-eye coordination and cognitive abilities. With the use of computers, we can improve our skills in areas such as typing, problem-solving, and critical thinking. This has been particularly beneficial for children, who are able to develop these skills at an early age.
    However, some experts argue that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. While it is true that excessive computer use can lead to a sedentary lifestyle, it is important to remember that it is up to each individual to balance their computer use with other activities.
    In conclusion, I believe that computers have a positive impact on society. They provide us with access to information, improve our cognitive abilities, and allow us to connect with people from all over the world. As long as we use computers in moderation, they can be a valuable tool for personal and professional growth.
    Thank you for considering my perspective on this important issue.

    Sincerely,
    Marcell O
    """

    result = tokenize_input(essay)
    token_ids = result.data["input_ids"]
    model = BertModel.from_pretrained("bert-base-uncased")
    embedding_matrix = model.embeddings.word_embeddings.weight
    vector_rep = embedding_matrix[token_ids]
    a = 1
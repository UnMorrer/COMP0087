# Base packages
import torch
import numpy as np
import os
from transformers import BertTokenizer, BertModel

# Custom packages
import src.models.bert as bert
import src.load.dataset_hf as load_data
import src.tokenization.general_hf_tokenizer as token_utils
import src.evaluation.utils as eval_utils

# Model trainer function
def model_trainer(
        torch_model_object,
        batch_size,
        epochs,
        tokenizer_object,
        tokenizer_model_object,
        optimizer_object,
        learning_rate,
        max_essay_tokens,
        model_save_name,
        training_device='cuda' if torch.cuda.is_available() else 'cpu',
        padding_strategy="right",
        truncation_strategy="end",
        checkpoints_enabled=True,
        epochs_per_checkpoint=10,
        model_save_dir="models",
        load_model_path=None
        ):
    """
    General function to train PyTorch models, do checkpointing
    and run evaluation.

    Inputs:
        torch_model_object - class inheriting after nn.Module: PyTorch
    class of model implementation. MUST HAVE:
            - .forward() method for running data through the model
            - .predict() method for obtaining evaluation predictions,
            above should NOT update model weights
            - .loss() function that returns model loss for batch
            batch_size - int: Batch size for model training & evaluation.
        epochs - int: Number of epochs for model training
    tokenizer_object - AutoTokenizer from transformers library.
    This tokenizer should be pre-initialized with model name
        tokenizer_model_object - AutoModel from transformers library.
    This model object should be initialized with model name. It must
    take input_ids from the tokenizer.
        optimizer_object - torch.optim object: Optimizer for the PyTorch model.
    It will be initialized with model parameters & learning rate.
        learning_rate - float: Learning rate for the model
        max_essay_tokens - int: The number of tokens (~words) that are
    processed for each essay. Essays containing fewer tokens than this
    will be padded according to "padding strategy". Essays containing
    more tokens than this will be truncated according to "truncation
    strategy".  
        model_save_name - str: Name for the model to save. Will be
    included in checkpoints & best model save.
        training_device - str: Name of training device. All data related
    to training (model, data etc.) will be moved to it.
        padding_strategy - str: padding direction. "right" pads the end
    of the sentence to reach max_length tokens. "left" pads the beginning
    of the sentence to reach max_tokens length.
        truncation_strategy - str: Truncation direction. "end" removes
    tokens from the end when sentences are too long. "beginning"
    removes tokens from the beginning when they are too long.
        checkpoints_enabled - bool: If True, enables model checkpointing
    (periodic saving of model state) while training. NOTE: They go in
    model save directory, previous checkpoints are not auto-deleted,
    and if the same model is trained again - the previous checkpoints
    are overwritten.
        epochs_per_checkpoint - int: Epochs between model saving. Only
    works when model checkpointing is enabled.
        model_save_dir - str: Model save directory. It is "started"
    from the current working directory.
        load_model_path - pythonPATH object: If this is given, the
    previously saved model data will be loaded in so training can
    be continued later on. If not given, training will start from
    a fresh model.

    Returns:
        None
    """

    # Data Loader
    data = load_data.read_in(
        sample=False
        )

    train_dataloader = torch.utils.data.DataLoader(
        data["train"].select_columns(["answer", "generated"]),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True # Ensure batch size is constant
    )

    eval_dataloader = torch.utils.data.DataLoader(
        data["validation"].select_columns(["answer", "generated"]),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Model-related things
    model = torch_model_object.to(training_device)

    # Resume model from checkpoint
    if load_model_path is not None:
        model.load_state_dict(torch.load(load_model_path))
        model.eval()

    optimizer = optimizer_object(model.parameters(), lr=learning_rate)

    # Tokenization
    tokenizer_model = tokenizer_model_object.to(training_device)

    best_eval_accuracy = 0
    for epoch in range(epochs):
        # Training loop
        training_loss = 0
        for batch in train_dataloader:
            # Reset gradients
            optimizer.zero_grad()

            # Tokenize input
            tokenized_batch = token_utils.tokenizer_function(
                batch["answer"],
                tokenizer=tokenizer_object,
                max_length=max_essay_tokens)
            input_vectors = token_utils.get_vector_representation(
                tokenized_batch["input_ids"],
                tokenizer_model
            ).to(training_device)

            # Shape of input_vectors:
            # <batch_size> x <num_tokens> x <encoding_size>
            outputs = model(input_vectors)

            # Convert classes to labels
            labels = batch["generated"].long() # 1 is Generated/Fake, 0 is Real
            labels = labels.to(training_device)
            loss = model.loss(outputs, labels)
            loss.backward()
            training_loss += loss.item()
            optimizer.step()

        # Validation loop
        correct = 0
        total = 0
        for batch in eval_dataloader:
            # Tokenize input
            tokenized_batch = token_utils.tokenizer_function(
                batch["answer"],
                tokenizer=tokenizer_object,
                max_length=max_essay_tokens,
                padding=padding_strategy,
                truncation=truncation_strategy)
            
            input_vectors = token_utils.get_vector_representation(
                tokenized_batch["input_ids"],
                tokenizer_model
            ).to(training_device)

            outputs = model.predict(input_vectors)
            # Gather correct predictions
            fake = outputs.detach().cpu().numpy()[:, 1]
            ground_truth = np.array(batch["generated"])

            # Save predictions
            correct += eval_utils.num_correct_predictions(fake, ground_truth)
            total += len(ground_truth)

        accuracy = correct / total
        print(f'Epoch {epoch}, training loss: {training_loss:.2f}, validation accuracy: {accuracy:.5f}')

        # Checkpoint saving
        if checkpoints_enabled and epoch % epochs_per_checkpoint == 0:
            model_save_path = os.path.join(
                os.getcwd(),
                model_save_dir, 
                model_save_name + f"_epoch{epoch}")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model to {model_save_path}")

        # Save best model so far
        if accuracy > best_eval_accuracy:
            best_eval_accuracy = accuracy
            print(f"New accuracy record: {accuracy:.5f}")
            model_save_path = os.path.join(
                os.getcwd(),
                model_save_dir, 
                model_save_name + "_best")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model to {model_save_path}")

# Example usage
if __name__ == "__main__":

    # Settings
    input_size = 768 # size of the BERT-encoded input
    hidden_size = 128
    num_classes = 2
    num_epochs = 50
    max_tokens = 512
    tokenizer_model_name = "bert-base-uncased"
    batch_size = 64
    epochs = 50
    lr = 0.1

    # Model-related things
    model = bert.RNNConnected(
        input_size,
        hidden_size,
        num_classes,
        batch_size,
        max_tokens
        )
    optimizer = torch.optim.Adam

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
    tokenizer_model = BertModel.from_pretrained(tokenizer_model_name)

    # Call model trainer function
    model_trainer(
        torch_model_object=model,
        batch_size=batch_size,
        epochs=num_epochs,
        tokenizer_object=tokenizer,
        tokenizer_model_object=tokenizer_model,
        optimizer_object=optimizer,
        learning_rate=lr,
        max_essay_tokens=max_tokens,
        model_save_name="example",
        training_device='cpu',
        padding_strategy="right",
        truncation_strategy="end",
        checkpoints_enabled=False,
        model_save_dir="models"
    )
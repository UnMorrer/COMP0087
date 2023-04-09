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

from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

average_vec = np.array([ 9.23821107e-02, -8.33334178e-02, -4.74294720e-05,  1.36092737e-01,
       -1.11753214e-02, -8.99242051e-03,  8.04364085e-02, -1.01534374e-01,
       -4.50804494e-02,  6.10840023e-01, -1.13657795e-01,  3.47111840e-03,
        1.00918554e-01, -1.08997978e-01, -8.33619833e-02, -1.21095352e-01,
        8.74321386e-02, -2.68112402e-02,  2.45265719e-02,  5.65140955e-02,
        4.17313576e-02, -6.88741356e-02, -2.08641350e-01, -1.06938221e-01,
        1.64321020e-01, -1.77382231e-02, -1.67867485e-02,  2.83149779e-02,
        7.04017058e-02, -5.70689030e-02, -2.60384772e-02, -1.84562773e-01,
        9.58825573e-02, -1.21241361e-01,  4.57528085e-01, -3.04208528e-02,
       -7.29278773e-02, -1.26595302e-02,  6.19916096e-02,  3.61088440e-02,
        4.24099900e-02,  8.48450214e-02,  4.51800488e-02, -1.89534217e-01,
       -2.90697701e-02, -2.75477953e-02, -6.76741451e-02, -1.16799802e-01,
        8.04973617e-02, -7.29644075e-02, -1.84061080e-02,  8.43591914e-02,
        1.34552084e-02, -2.69633867e-02,  2.98311207e-02, -1.27336472e-01,
        5.58404364e-02, -1.25067532e-01, -9.21262428e-02,  1.24383867e-01,
       -9.40691084e-02, -5.71042299e-02, -2.03706190e-01,  4.97911535e-02,
        6.30720183e-02,  2.55482495e-01, -8.01355094e-02, -5.61319739e-02,
       -1.47184422e-02,  5.67638092e-02, -3.55590247e-02, -1.98987611e-02,
       -1.78035647e-02, -5.30448332e-02,  1.07484728e-01, -2.28992887e-02,
       -1.24939620e-01, -1.05359562e-01,  1.01757482e-01,  2.64409799e-02,
        2.89462917e-02,  6.77810237e-02, -4.23418358e-03, -1.17672265e-01,
       -7.51884207e-02, -1.97265833e-03,  5.61311953e-02, -1.24113426e-01,
        6.38623396e-03, -2.51532830e-02,  7.35458583e-02, -1.23096690e-01,
        1.72608912e-01,  6.08213954e-02, -1.92695707e-02, -1.88152436e-02,
        1.81384861e-01, -2.18553897e-02, -3.61986384e-02,  2.79732585e-01,
        1.74815264e-02, -1.97855085e-02, -1.82405096e-02,  5.25284335e-02,
        3.15591730e-02,  9.57457535e-03, -1.37746677e-01, -6.69912770e-02,
        8.45852718e-02, -7.56566674e-02,  9.94375069e-03,  1.94309741e-01,
        5.62561825e-02,  8.98328051e-02, -1.06941871e-01, -1.08841866e-01,
        9.01055783e-02, -8.61963257e-02, -1.36264879e-02,  2.46704295e-01,
        3.48930620e-02,  3.60511467e-02, -2.84200534e-02,  2.15289779e-02,
        4.54886295e-02,  7.82944448e-03, -2.65242485e-03, -7.96690285e-02,
       -5.14672175e-02, -2.86113005e-03,  5.68292663e-03, -1.54272288e-01,
       -4.83916365e-02,  4.67422493e-02, -3.47001362e-03,  2.80795451e-02,
       -5.93680609e-03,  2.77160723e-02, -1.06079062e-03, -1.01578556e-01,
        4.95303757e-02, -3.87984067e-02, -7.66718090e-02, -1.13672040e-01,
        2.71457106e-01, -8.96425731e-03, -8.34825411e-02,  5.23360167e-03,
        1.34064376e-01, -3.42106121e-03, -3.61707896e-01,  8.81229155e-03,
       -6.46510422e-02,  1.38161015e-02, -2.79684812e-01, -5.25895841e-02,
        1.26042366e-01,  1.87356453e-02, -6.54089078e-02,  5.14815152e-02,
       -1.57615110e-01,  8.11077803e-02, -4.72714938e-02,  3.03989481e-02,
        8.42915028e-02, -7.92464241e-02,  1.39181735e-02, -4.13946956e-02,
        2.43484098e-02, -1.75853908e-01, -1.10805608e-01, -1.59475859e-02,
        3.63064855e-01, -7.12139457e-02, -3.16097378e-03, -5.10864705e-02,
        2.66881958e-02, -2.45828051e-02,  5.53645380e-02, -2.75467575e-01,
       -6.10495508e-02, -1.19413115e-01, -7.98880309e-02, -3.06054372e-02,
        3.02859209e-02,  1.12967618e-01,  2.34404504e-01, -6.50794571e-03,
       -2.78636590e-02, -8.21136236e-02,  1.53513877e-02, -1.41713664e-01,
        9.36345849e-03,  3.40282209e-02,  3.28226015e-02,  4.33810614e-02,
       -1.13316990e-01,  1.07238278e-01, -2.67703272e-02, -6.65315427e-03,
       -5.04602075e-01,  3.49752605e-02, -1.38338367e-02,  3.14966887e-02,
       -3.36701870e-02, -4.09896160e-03,  1.06694475e-01, -8.18595886e-02,
        2.03114236e-03,  9.92762670e-02,  4.53289272e-03,  9.06661227e-02,
       -6.53009862e-02,  1.25524821e-02, -1.24132395e-01, -6.20325767e-02,
       -3.20638567e-02, -9.58522335e-02,  2.00286992e-02, -3.53410811e-04,
       -2.33412266e-01,  9.28287506e-02,  8.48398060e-02, -2.44606696e-02,
        1.14548039e-02, -2.18968187e-02, -7.14387819e-02,  3.33529264e-02,
        3.87423225e-02,  4.57711853e-02, -1.25989290e-02, -7.83913117e-03,
       -1.12510622e-02,  5.84909283e-02, -1.40728071e-01, -5.62047884e-02,
       -5.34317531e-02, -4.50935774e-02,  6.88282102e-02, -6.75163120e-02,
        5.90525195e-02, -3.55978236e-02, -2.76978195e-01, -2.36738455e-02,
        3.50867957e-01,  1.67890079e-02, -1.27300307e-01,  1.04443036e-01,
        2.80100144e-02,  6.61000493e-04, -6.23330139e-02,  4.02932689e-02,
        4.96635288e-02,  3.40313725e-02, -3.12685817e-01,  9.29394439e-02,
        3.56945507e-02,  1.51343822e-01, -7.24697635e-02, -7.90918246e-02,
        3.09720095e-02,  1.14359319e-01, -1.02787912e-01, -7.39977211e-02,
        5.11922166e-02,  1.73084866e-02,  7.33908042e-02, -2.21398361e-02,
       -2.36990433e-02, -4.35655750e-02,  3.62321883e-02,  4.76490967e-02,
       -9.43688676e-02, -3.68515551e-02, -1.80413760e-02, -7.73986652e-02,
        6.67909145e-01, -1.68881137e-02, -2.71678269e-01, -6.27587438e-02,
        1.45923570e-01,  5.73927686e-02, -2.15114448e-02,  3.01603638e-02,
        1.08308103e-02, -1.88532144e-01,  6.71427473e-02, -1.21529855e-01,
       -3.74377295e-02,  1.40142916e-02,  4.98316698e-02,  1.68101802e-01,
        9.40177217e-02, -3.89945544e-02, -1.09528944e-01, -2.94609934e-01,
        2.17291317e-03,  2.18727857e-01,  1.81843743e-01, -6.70229569e-02])

# set up a function for OOV words, used in Glove defintion
def aver(tensor):
    return torch.Tensor(average_vec)

glove= GloVe(name='6B', dim=300, unk_init = aver)

tokenizer_basic = get_tokenizer('basic_english')

def glove_vectorizer(text_list, max_length, padding, truncation):

    # Determine if padding/truncation is required
    if max_length == 0:

        return sequence_of_vectors
    
    # Handle padding & truncation
    batch_size = 64
    

    vectorised_list_of_texts = []
    # Per-input based padding/truncation
    for i in range(len(text_list)):

        words_list = tokenizer_basic(text_list[i])
        sequence_of_vectors = glove.get_vecs_by_tokens(words_list)
        length = sequence_of_vectors.shape[0]

        if max_length == 0:
            vectorised_list_of_texts.append(sequence_of_vectors)

        if max_length == length:
            vectorised_list_of_texts.append(sequence_of_vectors)

        else:

            # Padding required
            if length < max_length:
                if padding.lower() == "right":
                    padded_sequence_of_vectors = torch.cat([sequence_of_vectors, torch.zeros(size = (max_length - length,300))], dim = 0)
                    vectorised_list_of_texts.append(padded_sequence_of_vectors)

                elif padding.lower() == "left":
                    padded_sequence_of_vectors = torch.cat([torch.zeros(size = (max_length - length,300)), sequence_of_vectors], dim = 0)
                    vectorised_list_of_texts.append(padded_sequence_of_vectors)

                else:
                    raise ValueError(f"Invalid value for padding: {padding}. Please use 'left' or 'right' instead.")

            # Truncation required
            elif length > max_length:
                if truncation.lower() == "end":
                    truncated_sequence_of_vectors = sequence_of_vectors[:max_length,:]
                    vectorised_list_of_texts.append(truncated_sequence_of_vectors)

                elif truncation.lower() == "beginning":
                    truncated_sequence_of_vectors = sequence_of_vectors[(length -max_length):,:]
                    vectorised_list_of_texts.append(truncated_sequence_of_vectors)
            
                else:
                    raise ValueError(f"Invalid value for truncation: {truncation}. Please use 'end' or 'beginning' instead.")
        
    # Return result
    # <batch_size> x <num_tokens> x <encoding_size>

    unsqueezed_list = [torch.unsqueeze(seq_of_vecs, 0) for seq_of_vecs in vectorised_list_of_texts]
    return torch.cat(unsqueezed_list, dim = 0)




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
        print("This is epoch #" + str(epoch))
        # Training loop
        training_loss = 0
        for batch in train_dataloader:
            # Reset gradients
            optimizer.zero_grad()

            # Tokenize input
            input_vectors = glove_vectorizer(batch["answer"], max_essay_tokens, padding_strategy, truncation_strategy).to(training_device)

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
            input_vectors = glove_vectorizer(batch["answer"], max_essay_tokens, padding_strategy, truncation_strategy).to(training_device)

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
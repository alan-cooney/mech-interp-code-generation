
from datasets import load_dataset
from torch.utils.data import DataLoader
from easy_transformer import loading_from_pretrained
from torchtyping import TensorType
from tqdm import tqdm
from typing import List, Optional
import torch
from easy_transformer import EasyTransformer
import numpy as np
from pathlib import Path
from os import makedirs
import re


def get_dataset(
    dataset_name: Optional[str] = "NeelNanda/code-tokenized"
) -> DataLoader:
    """Get a dataset from the HuggingFace datasets library

    Note this will take a few minutes to run for a reasonable sized dataset.

    Args:
        dataset_name: Dataset name from HuggingFace.

    Returns:
        Iterable of the dataset. Each batch is a dict with one
        key ("tokens") that contains a tensor of size (number_prompts,
        number_tokens).
    """
    # Load the dataset (caches locally as a file)
    ds = load_dataset(dataset_name, split="train")

    # Set the format as a PyTorch Tensor of size (position)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds.set_format(type="torch", columns=["tokens"], device=device)

    # Batch
    dataloader = DataLoader(ds, shuffle=False, batch_size=10)

    return dataloader


def run_batch(
    model: EasyTransformer,
    data: TensorType["batch_item", "token_position"]
) -> TensorType["batch_item", "token_position", "vocab_size"]:
    """Run a batch of data through the model and return the correct log
    probabilities

    Args:
        model (EasyTransformer): Model
        data (TensorType["batch_item", "token_position"]): Data

    Returns:
        Correct log probabilities
    """

    # Get the logits
    logits: TensorType["batch_item", "position", "vocab"] = model(data)
    # Remove the last token (as we don't have a "correct token" to compare
    # it against)
    logits_except_last = logits[:, :-1, :]

    # Get the log probs
    log_probs: TensorType["batch_item", "position", "vocab"] \
        = logits_except_last.log_softmax(dim=-1)

    # Get the correct tokens
    correct_tokens: TensorType["batch_item", "position"] = data[:, 1:]

    # Get the correct log probs
    correct_log_probs: TensorType["batch_item", "vocab"] \
        = log_probs.gather(dim=-1, index=correct_tokens.unsqueeze(-1)).squeeze(-1)
    return correct_log_probs


def save_batch_results(results: List[np.ndarray], model_name: str):
    results_array = np.stack(results, axis=0)

    # Get the checkpoints path
    checkpoint_dir = Path(__file__).parent / ".checkpoints" / model_name
    file_name = re.sub('[^A-Za-z0-9]+', '', model_name)
    checkpoint_path = (checkpoint_dir / file_name).with_suffix(".npy")
    makedirs(checkpoint_dir, exist_ok=True)

    np.save(checkpoint_path, results_array)


def run_model(
    model_name: loading_from_pretrained.OFFICIAL_MODEL_NAMES
) -> np.ndarray:
    """Get the correct log probabilities for a model

    Args:
        model_name: Model name

    Returns:
        Correct log probabilities for all the prompts & tokens in the dataset
    """
    model = EasyTransformer.from_pretrained(model_name)
    batches = get_dataset()

    # Initialise results
    results = []

    # Print the model we're running
    print(f"Running {model_name}")

    # For each batch
    batch_number = 0
    for batch_data in tqdm(batches):
        data: TensorType["batch_item", "token_position"] = batch_data["tokens"]
        batch_results = run_batch(model, data)
        results.append(batch_results.detach().cpu().numpy())

        # Save every x batches
        batch_number += 1
        if batch_number % 100 == 0:
            save_batch_results(results, model_name)

    save_batch_results(results, model_name)
    return results.stack(axis=0)

import torch
import h5py
import os
import glob
from torch.utils.data import Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence


class H5BrainDataset(Dataset):
    """
    PyTorch Dataset for loading the Brain-to-Text HDF5 data.

    This class loads one HDF5 file (representing one session) and
    makes its trials accessible for training.
    """

    def __init__(self, h5_path, tokenizer, max_text_len=128):
        """
        Args:
            h5_path (str): Path to a single .hdf5 file.
            tokenizer (Tokenizer): A Hugging Face tokenizer (e.g., from GPT2).
            max_text_len (int): The maximum length for tokenizing text labels.
        """
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.file = h5py.File(h5_path, "r")

        # Sort groups numerically
        self.trial_keys = sorted(self.file.keys(), key=lambda k: int(k.split("_")[-1]))

    def __len__(self):
        return len(self.trial_keys)

    def __getitem__(self, idx):
        key = self.trial_keys[idx]
        trial_group = self.file[key]

        neural_data_raw = trial_group["input_features"][:]
        neural_features = torch.tensor(neural_data_raw, dtype=torch.float32)

        text_str = trial_group.attrs["sentence_label"]

        tokenized_text = self.tokenizer(
            text_str,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        # Remove batch dimension
        labels = tokenized_text.input_ids.squeeze(0)

        return {"neural_features": neural_features, "labels": labels}

    def close(self):
        self.file.close()


class DataCollator:
    """
    Pad data for training
    """

    def __init__(self, tokenizer):
        # Padding labels
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        neural_list = [sample["neural_features"] for sample in batch]
        padded_neural = pad_sequence(neural_list, batch_first=True, padding_value=0.0)

        # Create attention mask
        neural_lengths = [len(n) for n in neural_list]
        encoder_attention_mask = torch.zeros(
            padded_neural.shape[0],  # batch_size
            padded_neural.shape[1],  # max_seq_len
            dtype=torch.long,
        )
        for i, length in enumerate(neural_lengths):
            encoder_attention_mask[i, :length] = 1

        labels_list = [sample["labels"] for sample in batch]
        padded_labels = pad_sequence(
            labels_list, batch_first=True, padding_value=self.pad_token_id
        )

        # Pytorch default ignore index is -100.
        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
        padded_labels[padded_labels == self.pad_token_id] = -100

        return {
            "inputs_embeds": padded_neural,  # For the encoder
            "attention_mask": encoder_attention_mask,  # For the encoder
            "labels": padded_labels,  # For the decoder
        }


def get_brain_dataset(data_root, split, tokenizer):
    """
    Finds all HDF5 files for a split, loads them into H5BrainDataset,
    and returns a single ConcatDataset.
    """
    print(f"Loading {split} data from {data_root}...")

    pattern = os.path.join(
        data_root,
        "t15_copyTask_neuralData",
        "hdf5_data_final",
        "*",
        f"data_{split}.hdf5",
    )
    files = glob.glob(pattern)

    if not files:
        raise ValueError(f"Warning: No {split} files found")

    print(f"Found {len(files)} files for split '{split}'.")

    datasets = [H5BrainDataset(f, tokenizer) for f in files]

    combined_dataset = ConcatDataset(datasets)

    print(f"Total examples in '{split}': {len(combined_dataset)}")
    return combined_dataset

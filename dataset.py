import torch
import h5py
from torch.utils.data import Dataset
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
        
        self.file = h5py.File(h5_path, 'r')
        
        # Sort groups numerically
        self.trial_keys = sorted(self.file.keys(), key=int)
        
        print(f"Loaded {h5_path} with {len(self.trial_keys)} trials.")

    def __len__(self):
        return len(self.trial_keys)

    def __getitem__(self, idx):
        key = self.trial_keys[idx]
        trial_group = self.file[key]
        
        neural_data_raw = trial_group['input_features'][:]
        
        # Transpose from (512, T) to (T, 512)
        neural_features = torch.tensor(neural_data_raw, dtype=torch.float32).T
        
        text_bytes = trial_group.attrs['sentence_label']
        text_str = text_bytes.decode('utf-8')
        
        tokenized_text = self.tokenizer(
            text_str, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        labels = tokenized_text.input_ids.squeeze(0)
        
        return {
            "neural_features": neural_features,
            "labels": labels
        }

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
        neural_list = [sample['neural_features'] for sample in batch]
        padded_neural = pad_sequence(
            neural_list, 
            batch_first=True, 
            padding_value=0.0
        )
        
        # Create attention mask
        neural_lengths = [len(n) for n in neural_list]
        encoder_attention_mask = torch.zeros(
            padded_neural.shape[0], # batch_size
            padded_neural.shape[1], # max_seq_len
            dtype=torch.long
        )
        for i, length in enumerate(neural_lengths):
            encoder_attention_mask[i, :length] = 1
        
        labels_list = [sample['labels'] for sample in batch]
        padded_labels = pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=self.pad_token_id
        )
        
        # Pytorch default ignore index is -100. 
        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
        padded_labels[padded_labels == self.pad_token_id] = -100

        return {
            "inputs_embeds": padded_neural,        # For the encoder
            "attention_mask": encoder_attention_mask, # For the encoder
            "labels": padded_labels                # For the decoder
        }
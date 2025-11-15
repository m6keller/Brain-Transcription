import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# --- 1. Define the Phoneme Vocabulary ---
# (You'll need a real phoneme tokenizer for this. 
# You can find standard English phoneme maps, like ARPABET)
# For this example, we'll mock one.
# IMPORTANT: The 0-index is RESERVED for the CTC 'blank' token.
PHONEME_MAP = {
    "_": 0,  # BLANK token
    " ": 1,  # Word separator
    "A": 2,
    "B": 3,
    "C": 4,
    "D": 5,
    "E": 6,
    "F": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "J": 11,
    "K": 12,
    "L": 13,
    "M": 14,
    "N": 15,
    "O": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "U": 22,
    "V": 23,
    "W": 24,
    "X": 25,
    "Y": 26,
    "Z": 27,
}
VOCAB_SIZE = len(PHONEME_MAP) # e.g., 28 in this mock example


# --- 2. Define the Neural Encoder Model ---

class TemporalBlock(nn.Module):
    """TCN Temporal Block"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # --- THIS IS THE MISSING LINE ---
        self.padding = padding
        # ---------------------------------

        # Define layers individually
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # self.net = ... (This line should still be removed)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding].contiguous() # Apply chomp (slicing)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.padding].contiguous() # Apply chomp (slicing)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # The residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: (Batch, Features, Time)
        # TCN expects (Batch, Features, Time)
        return self.network(x)


class PhonemeEncoder(nn.Module):
    def __init__(self, input_features_dim=512, tcn_hidden_dim=256, phoneme_vocab_size=28):
        super().__init__()
        
        # This could be your NeuralFeatureEncoder from before
        self.projection = nn.Linear(input_features_dim, tcn_hidden_dim)
        
        # This is the TCN encoder
        self.tcn = TCNEncoder(tcn_hidden_dim, num_channels=[tcn_hidden_dim] * 4) # 4 TCN layers
        
        # This is your "MLP" head to classify phonemes
        self.phoneme_head = nn.Linear(tcn_hidden_dim, phoneme_vocab_size)

    def forward(self, inputs_embeds):
        # inputs_embeds: (Batch, Time, 512)
        
        # 1. Project to hidden dim
        x = self.projection(inputs_embeds) # (Batch, Time, 256)
        
        # 2. TCN expects (Batch, Features, Time)
        x = x.permute(0, 2, 1)
        x = self.tcn(x) # (Batch, 256, Time)
        
        # 3. Classifier head expects (Batch, Time, Features)
        x = x.permute(0, 2, 1)
        logits = self.phoneme_head(x) # (Batch, Time, 28)
        
        # 4. CTC Loss expects log probabilities
        # (Time, Batch, Classes)
        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        
        return log_probs

# --- 3. Example Training Step (THIS IS KEY) ---

# Assume:
# - `model` is an instance of PhonemeEncoder
# - `optimizer` is set up
# - `batch` is a dict from your DataLoader

# Mock data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhonemeEncoder(phoneme_vocab_size=VOCAB_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# (Batch, Time, Features)
inputs_embeds = torch.randn(4, 100, 512).to(device)
# List of real lengths
input_lengths = torch.tensor([100, 80, 90, 75]).to(device)

# --- Target preparation (the tricky part) ---
# Your dataloader needs to phonemize the text and pad it
# "HELLO" -> [H, E, L, L, O] -> [8, 5, 12, 12, 15]
targets_list = [
    torch.tensor([2, 3, 4, 4, 5]),
    torch.tensor([6, 7, 3]),
    torch.tensor([2, 8, 9, 10, 11, 12]),
    torch.tensor([13, 14, 15, 4])
]
# We smash them into one long tensor
targets = torch.cat(targets_list).to(device)
# We tell CTC the length of each *target*
target_lengths = torch.tensor([len(t) for t in targets_list]).to(device)

# --- Define CTC Loss ---
# `blank=0` tells it to use the 0-index as the blank token
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

# --- Training Step ---
optimizer.zero_grad()

# (Time, Batch, Classes)
log_probs = model(inputs_embeds)

# Calculate loss
loss = ctc_loss(
    log_probs,          # (Time, Batch, Classes)
    targets,            # (Total_Target_Length)
    input_lengths,      # (Batch)
    target_lengths      # (Batch)
)

loss.backward()
optimizer.step()

print(f"CTC Loss: {loss.item()}")
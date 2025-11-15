import kenlm
import numpy as np
from pyctcdecode import build_ctcdecoder

# --- 1. Teammates Train the Language Model ---
# This is a one-time step, run in the terminal.
# Assume they've collected all training sentences into `corpus.txt`.

# $ lmplz -o 5 < corpus.txt > 5gram.arpa
# (This creates the '5gram.arpa' LM file)


# --- 2. Teammates Build the Decoder Script ---

# The phoneme labels MUST match your Stage 1 model
PHONEME_LABELS = [
    "_", " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", 
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", 
    "T", "U", "V", "W", "X", "Y", "Z"
]

# Path to the LM file they just trained
lm_path = "5gram.arpa"

# Instantiate the decoder
decoder = build_ctcdecoder(
    labels=PHONEME_LABELS,
    kenlm_model_path=lm_path,
    # You can add more text files for words the LM might
    # not know, but are in the phoneme vocab
    # unigrams=["list", "of", "words"] 
)

print("CPU-based decoder built successfully!")

# --- 3. Teammates Run Inference ---
# They load the output from YOUR model.
# Let's mock `log_probs` (shape: Time, Batch, Classes)
# In a real run, you'd do:
# log_probs = np.load("model_output_for_batch_0.npy")
# Note: pyctcdecode wants (Batch, Time, Classes) or (Time, Classes)
mock_log_probs_TBC = np.random.rand(100, 1, 28).astype(np.float32) # (Time, Batch, Classes)
mock_log_probs_BTC = mock_log_probs_TBC.transpose(1, 0, 2) # (Batch, Time, Classes)


print("\nRunning decoding...")

# Run the beam search! This is the magic.
# This combines the "sound-alike" (log_probs) with the
# "make-sense" (LM) score.
# This is 100% on the CPU.
decoded_text = decoder.decode(mock_log_probs_BTC)

print(f"Decoded Text: {decoded_text[0]}")

# They can also get more complex outputs
# result = decoder.decode_beams(mock_log_probs_BTC[0])
# print(f"Top beam: {result[0][0]}")
# print(f"Score: {result[0][1]}")
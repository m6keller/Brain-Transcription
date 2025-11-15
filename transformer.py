import os
import torch
import h5py
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    EncoderDecoderModel,
)

from dataset import DataCollator, H5BrainDataset


class NeuralFeatureEncoder(nn.Module):
    """
    Process input neural features into embeddings suitable for BERT.
    """

    def __init__(self, config, input_features_dim=512):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.projection = nn.Linear(input_features_dim, self.hidden_size)

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, self.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    # HACK: Must match BertEmbeddings.forward signature
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        """
        This signature MUST match the base class (BertEmbeddings.forward).
        - `input_ids` will be None (we pass `inputs_embeds` instead).
        - `inputs_embeds` will be our (batch, seq_len, 512) neural tensor.
        """

        if inputs_embeds is None:
            raise ValueError(
                "NeuralFeatureEncoder expects 'inputs_embeds', not 'input_ids'"
            )

        # `inputs_embeds` is our `neural_features` tensor
        # shape: [batch_size, seq_len, 512]

        # 1. Project features from 512 -> 768
        projected_features = self.projection(inputs_embeds)

        # 2. Add position embeddings
        seq_length = inputs_embeds.size(1)

        if position_ids is None:
            # Create default position_ids
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        pos_embeddings = self.position_embeddings(position_ids)

        # Combine projected features and position embeddings
        embeddings = projected_features + pos_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Return the final (batch, seq_len, 768) tensor
        return embeddings


def build_model():
    """
    Initializes the Encoder-Decoder model.

    TODO: Allow for different encoder/decoder choices. (gpt2 decoder and ber-base-uncased are hardcoded for now)
    """

    decoder = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    decoder_config = AutoConfig.from_pretrained("gpt2")

    decoder_config.add_cross_attention = True
    decoder_config.pad_token_id = tokenizer.pad_token_id

    # Load decoder from modeified config
    decoder = AutoModelForCausalLM.from_pretrained("gpt2", config=decoder_config)

    decoder.config.pad_token_id = decoder.config.eos_token_id

    encoder = AutoModel.from_pretrained("bert-base-uncased")

    # HACK: Replace the embedding layer with custom encoder
    encoder.embeddings = NeuralFeatureEncoder(encoder.config, input_features_dim=512)

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    model.config.is_encoder_decoder = True
    model.config.decoder.add_cross_attention = True

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model built successfully!")
    print(f"  Encoder: {encoder.config.model_type}")
    print(f"  Decoder: {decoder.config.model_type}")

    return model, tokenizer


def main():
    """
    Main function demonstrating a proper training step
    using the DataLoader and DataCollator.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build model and tokenizer
    model, tokenizer = build_model()
    model.to(device)

    # 2. Create Dummy Data
    DUMMY_H5_PATH = "./dummy_brain_data.h5"

    try:
        with h5py.File(DUMMY_H5_PATH, "w") as f:
            print("Creating dummy HDF5 file for example...")
            for i in range(10):  # Create 10 dummy trials
                g = f.create_group(str(i))
                seq_len = 50 + i * 5  # Variable sequence length

                # Neural data: (features, time) -> (512, T)
                dummy_neural = np.random.rand(512, seq_len).astype(np.float32)
                g.create_dataset("input_features", data=dummy_neural)

                # Text label: as a bytes attribute
                dummy_text = f"this is example sentence {i}"
                g.attrs["sentence_label"] = np.bytes_(dummy_text)

        print("Dummy file created successfully.")
    except Exception as e:
        print(f"Could not create dummy file. {e}")
        return  # Exit if we can't create the file

    # 3. Set up Dataset, Collator, and DataLoader
    dataset = H5BrainDataset(DUMMY_H5_PATH, tokenizer)
    data_collator = DataCollator(tokenizer)

    # This now handles all the padding and batching!
    data_loader = DataLoader(
        dataset,
        batch_size=4,  # You can change this
        collate_fn=data_collator,
        shuffle=True,
    )

    # 4. Set up optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("\n--- Starting Training Loop (Example) ---")

    # 5. Run one training step
    model.train()

    # Get one batch from the data_loader
    # A real loop would be: `for batch in data_loader:`
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        print("DataLoader is empty (this shouldn't happen).")
        return

    # Move all tensors in the batch to the device
    inputs_embeds = batch["inputs_embeds"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print(f"Batch input (neural) shape: {inputs_embeds.shape}")
    print(f"Batch attention mask shape: {attention_mask.shape}")
    print(f"Batch labels shape: {labels.shape}")

    # Forward pass
    optimizer.zero_grad()  # Clear old gradients

    outputs = model(
        inputs_embeds=inputs_embeds,  # Padded neural data
        attention_mask=attention_mask,  # The mask from the collator
        labels=labels,  # Padded labels (with -100)
    )

    loss = outputs.loss
    print(f"\nForward Pass Successful!")
    print(f"Batch Loss: {loss.item()}")

    # 6. Backward pass
    loss.backward()
    optimizer.step()

    print("Backward pass (training step) successful!")

    # 7. Generation (Inference)
    print("\n--- Starting Generation (Example) ---")
    model.eval()
    with torch.no_grad():
        # Use the same batch's inputs_embeds and attention_mask
        # for this generation example
        generated_ids = model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_length=50
        )

        # Decode the first sentence in the batch
        decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"Generated text (from 1st item in batch):")
        print(f"-> {decoded_text}")

    # 8. Clean up
    dataset.close()
    if "DUMMY_H5_PATH" in locals() and "dummy" in DUMMY_H5_PATH:
        os.remove(DUMMY_H5_PATH)
        print("\nCleaned up dummy file.")


if __name__ == "__main__":
    main()

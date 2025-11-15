import torch
import os
import numpy as np
import json
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from model import build_model

from dataset import DataCollator, get_brain_dataset

# --- Configuration ---
DATA_ROOT = "/scratch/m6keller/brain-to-text"
MODEL_OUTPUT_DIR = "./brain_model_v1"
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
# ---------------------


def run_validation(model, val_loader, device):
    """Run one epoch of validation."""
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            # Move batch to device
            inputs_embeds = batch["inputs_embeds"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_val_loss += outputs.loss.item()

    return total_val_loss / len(val_loader)


def train():
    """Main training and validation loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build Model and Tokenizer
    model, tokenizer = build_model()
    model.to(device)

    # 2. Create DataCollator
    data_collator = DataCollator(tokenizer=tokenizer)

    # 3. Load Datasets (This is now much simpler)
    try:
        train_dataset = get_brain_dataset(DATA_ROOT, "train", tokenizer)
    except FileNotFoundError as e:
        print(e)
        return  # Exit if no training data

    val_dataset = get_brain_dataset(DATA_ROOT, "val", tokenizer)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE * 2,  # Use larger batch size for val
            collate_fn=data_collator,
            shuffle=False,
        )
        print(f"Total validation steps: {len(val_loader)}")

    print(f"Total training steps per epoch: {len(train_loader)}")

    # 5. Set up Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 6. Training Loop
    best_val_loss = float("inf")
    training_stats = []

    # Ensure output directory exists
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        # --- Training ---
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch in progress_bar:
            # Move batch to device
            inputs_embeds = batch["inputs_embeds"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        if val_loader:
            avg_val_loss = run_validation(model, val_loader, device)
            print(
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
            )
        else:
            avg_val_loss = -1.0  # No validation
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} (No validation)")

        # Save stats
        stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        training_stats.append(stats)

        # --- Save Best Model ---
        if val_loader and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best val loss. Saving model...")
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        elif not val_loader:
            # If no validation, just save the model every epoch
            print("Saving model checkpoint...")
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

        # Save stats to a file
        with open(os.path.join(MODEL_OUTPUT_DIR, "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=2)

    print("\n--- Training complete! ---")
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model and stats saved to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    train()

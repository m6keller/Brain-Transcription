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
from utils import DATA_ROOT

# --- Configuration ---
NUM_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
# ---------------------


def run_validation(model, val_loader, device):
    """Run one epoch of validation."""
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            inputs_embeds = batch["inputs_embeds"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_val_loss += outputs.loss.item()

    return total_val_loss / len(val_loader)


def freeze_model_layers(model, last_layers_to_train=1):
    """
    Freezes all layers except the classification head and the last N transformer layers.
    """
    print(f"\n--- Fine-tune Mode: Freezing early layers ---")
    
    for param in model.parameters():
        param.requires_grad = False

    # 2. Detect the deepest layer index
    max_layer_idx = -1
    for name, _ in model.named_parameters():
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                idx = int(part)
                if idx > max_layer_idx:
                    max_layer_idx = idx
    
    print(f"Detected max layer index: {max_layer_idx}")

    # Unfreeze Head and Last N layers
    trainable_count = 0
    for name, param in model.named_parameters():
        should_train = False
        
        # Always train prediction heads/classifiers
        if any(keyword in name for keyword in ['classifier', 'head', 'output', 'linear', 'projector']):
            should_train = True
            
        if max_layer_idx != -1:
            parts = name.split('.')
            for part in parts:
                if part.isdigit() and int(part) > (max_layer_idx - last_layers_to_train):
                    should_train = True
                    break
        
        if should_train:
            param.requires_grad = True
            trainable_count += 1

    print(f"Unfrozen {trainable_count} parameter groups (Heads + Last {last_layers_to_train} layers).")


def train(model_output_dir: os.PathLike, finetune: bool = False):
    """Main training and validation loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = build_model()
    model.to(device)

    if finetune:
        freeze_model_layers(model, last_layers_to_train=2)

    data_collator = DataCollator(tokenizer=tokenizer)

    try:
        train_dataset = get_brain_dataset(DATA_ROOT, "train", tokenizer)
    except FileNotFoundError as e:
        print(e)
        return  

    val_dataset = get_brain_dataset(DATA_ROOT, "val", tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE * 2,  
            collate_fn=data_collator,
            shuffle=False,
        )
        print(f"Total validation steps: {len(val_loader)}")

    print(f"Total training steps per epoch: {len(train_loader)}")

    # Important: Filter parameters so optimizer only sees trainable ones
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_val_loss = float("inf")
    training_stats = []

    os.makedirs(model_output_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

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

        stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        training_stats.append(stats)

        if val_loader and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best val loss. Saving model...")
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
        elif not val_loader:
            print("Saving model checkpoint...")
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)

        with open(os.path.join(model_output_dir, "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=2)

    print("\n--- Training complete! ---")
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model and stats saved to {model_output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the brain transcription model.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./brain_model_v1",
        help="Directory to save the trained model and tokenizer.",
    )

    parser.add_argument(
        "--finetune",
        action="store_true",
        help="If set, only train the last layer and the head.",
    )
    
    args = parser.parse_args()
    
    train(model_output_dir=args.output_dir, finetune=args.finetune)
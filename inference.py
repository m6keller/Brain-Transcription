import torch
import argparse
from pathlib import Path

from dataset import get_brain_dataset, DataCollator
from model import build_model
from utils import DATA_ROOT

def main(path_to_state_dict: Path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = build_model(state_dict_path=path_to_state_dict)
    model.to(device)
    model.eval()

    test_dataset = get_brain_dataset(DATA_ROOT, "val", tokenizer)

    sample_index = 0
    raw_sample = test_dataset[sample_index]
    
    print(f"\n--- Inference on Sample Index {sample_index} ---")

    collator = DataCollator(tokenizer)
    
    batch = collator([raw_sample])

    inputs_embeds = batch["inputs_embeds"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print("Generating text from neural signals...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    ground_truth_text = tokenizer.decode(raw_sample["labels"], skip_special_tokens=True)

    print("-" * 60)
    print(f"Neural Shape:  {inputs_embeds.shape}")
    print(f"Ground Truth:  {ground_truth_text}")
    print("-" * 60)
    print(f"Prediction:    {predicted_text}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the brain transcription model.")
    parser.add_argument(
        "--state-dict",
        type=str,
        default=None,
        help="Path to the model state dictionary or directory.",
    )
    
    args = parser.parse_args()
    main(path_to_state_dict=args.state_dict)
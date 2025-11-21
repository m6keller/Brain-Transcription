import torch
import argparse
import jiwer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_brain_dataset, DataCollator
from model import build_model
from utils import DATA_ROOT

BATCH_SIZE = 8

def compute_metrics(predictions, references):
    """
    Computes Word Error Rate (WER).
    """
    
    wer = jiwer.wer(references, predictions)
    return wer

def main(state_dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model...")
    model, tokenizer = build_model(state_dict_path=state_dict_path)
    model.to(device)
    model.eval()

    val_dataset = get_brain_dataset(DATA_ROOT, "val", tokenizer)
    
    collate_fn = DataCollator(tokenizer)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=False
    )

    print(f"\nStarting evaluation on {len(val_dataset)} validation examples...")

    total_loss = 0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating")
        
        for batch in progress_bar:
            inputs_embeds = batch["inputs_embeds"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()

            generated_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=1,
                repetition_penalty=1.2,
                early_stopping=True,
            )

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            labels_for_decoding = labels.clone()
            labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(labels_for_decoding, skip_special_tokens=True)

            all_predictions.extend(pred_texts)
            all_references.extend(ref_texts)

            progress_bar.set_postfix({"loss": outputs.loss.item()})

    avg_loss = total_loss / len(val_loader)
    wer_score = compute_metrics(all_predictions, all_references)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Word Error Rate (WER): {wer_score:.4f}")
    
    print("\n" + "="*50)
    print("QUALITATIVE EXAMPLES (First 5)")
    print("="*50)
    
    for i in range(min(5, len(all_predictions))):
        print(f"Example {i+1}:")
        print(f"  REF:  {all_references[i]}")
        print(f"  PRED: {all_predictions[i]}")
        print("-" * 30)

    output_file = "validation_predictions.txt"
    with open(output_file, "w") as f:
        for ref, pred in zip(all_references, all_predictions):
            f.write(f"REF:  {ref}\nPRED: {pred}\n\n")
    print(f"\nFull predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dict", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    main(args.state_dict)
import torch
from tqdm import tqdm

def parse_outputs(model_output_texts, taxonomy):
    parsed = []
    for text in model_output_texts:
        y_pred = taxonomy.parse_response(text)
        parsed.append(y_pred)

    return parsed

@torch.no_grad
def evaluate(model, tokenizer, dataloader, taxonomy, **generate_kwargs):
    model.eval()
    true_positives, parsed = 0, 0

    total_samples = 0

    for i, batch in enumerate(tqdm(dataloader), start=1):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        y_trues = batch["y_true"]

        # Generate model predictions
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        y_preds = parse_outputs(decoded, taxonomy)

        # Grade predictions
        tps, p = taxonomy.grade_outputs(y_preds, y_trues)
        batch_size = len(y_trues)
        true_positives += tps
        parsed += p
        total_samples += batch_size

        # Running means
        running_accuracy = true_positives / total_samples
        running_parse_rate = parsed / total_samples

        print(f"[Batch {i}/{len(dataloader)}] "
              f"Running Accuracy: {running_accuracy:.4f}, "
              f"Running Parse Rate: {running_parse_rate:.4f}")

    final_accuracy = true_positives / total_samples
    final_parse_rate = parsed / total_samples

    print(f"\nFinal Accuracy: {final_accuracy:.4f}, Final Parse Rate: {final_parse_rate:.4f}")
    return {'accuracy': final_accuracy, 'parse_rate': final_parse_rate}
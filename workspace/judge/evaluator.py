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
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        y_trues = batch["y_true"]

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        y_preds = parse_outputs(decoded, taxonomy)
        tps, p = taxonomy.grade_outputs(y_preds, y_trues)
        true_positives += tps
        parsed += p
    
    accuracy, parse_rate = true_positives / len(dataloader.dataset), parsed / len(dataloader.dataset)
    return {'accuracy' : accuracy, 'parse_rate' : parse_rate}

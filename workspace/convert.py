import pandas as pd
with open('precomputed_logprob_test.jsonl', 'r') as file:
    frame = pd.read_json(file, lines=True)
frame.to_parquet('tmp.parquet')


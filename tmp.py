from verl.protocol import DataProto
from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
b = DataProto.load_from_disk('/u/rfechner/verl/batch.pt')
a = DataProto.load_from_disk('/u/rfechner/verl/out.pt')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Example chat messages
messages = [{'role': 'user', 'content': "What's 2+2?"}]
messages_full = [
    {'role': 'user', 'content': "What's 2+2?"},
    {'role': 'assistant', 'content': "4!"}
]

# print("Raw chat messages:")
# print(tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False))

# Apply chat template for generation
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate model response
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)
print(outputs.shape)

# Decode model output
response = tokenizer.decode(outputs[0])
print("Model response:")
print(response)

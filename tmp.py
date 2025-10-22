
# from verl.protocol import DataProto
from transformers import AutoTokenizer, AutoModelForCausalLM

models = {
    'llama3-small' : "meta-llama/Llama-3.2-1B-Instruct",
    'llama3-medium' : "meta-llama/Llama-3.2-3B-Instruct",
    'llama3-large' : "meta-llama/Llama-3.1-8B-Instruct",
    'qwen3-small' : "Qwen/Qwen3-0.6B",
    'qwen3-medium' : "Qwen/Qwen3-4B",
    'qwen3-large' : "Qwen/Qwen3-8B",
    'moxin' : "moxin-org/Moxin-7B-Instruct",
    'r1-qwen-medium' : "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    'r1-qwen-large' : "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
    'r1-llama-large' : "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

for model in models.values():
    try:
        print("Processing model: ", model)
        t = AutoTokenizer.from_pretrained(model)
        m = AutoModelForCausalLM.from_pretrained(model)
        assert t.chat_template is not None
        print("Processed model: ", model, " without errors")
    except (Exception, AssertionError) as e:
        print("Model: ", model, f" encountered error: {e}")
    

# b = DataProto.load_from_disk('/u/rfechner/verl/batch.pt')
# a = DataProto.load_from_disk('/u/rfechner/verl/out.pt')

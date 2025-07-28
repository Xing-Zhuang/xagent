import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate(model_id):
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16)
    model.to(device)


    test_texts = {
        'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
        'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
    }
    instruction = test_texts['instruction']
    input_value = test_texts['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]
 
    

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\n\n======{model_id}======")
    print(response)



generate("Qwen/Qwen2-1.5B")
generate("./qwen2-finetune-final")
 
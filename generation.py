import os
import torch
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import AutoTokenizer
from utils import load_prompts, evaluation
import Baseline.joy_data_loader as loader

device = 0 if torch.cuda.is_available() else 'cpu'

def generate(model_path, prompts, seed=1234567):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # output_min_length = 4
    # output_max_length = 128
    # output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "max_length":100,
        "top_k": 0.0,
        "top_p": 0.9,
        "temperature": 1.0,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "num_return_sequences": 500,
        # "pad_token_id": tokenizer.eos_token_id,
    }
    # gen_len = output_length_sampler()
    # generation_kwargs["max_new_tokens"] = gen_len

    decoded_responses = []
    for idx, p in enumerate(prompts):
        encoded_prompts = torch.tensor(tokenizer.encode(p)).long().to(device) 
        response = model.generate(input_ids=encoded_prompts.unsqueeze(0), **generation_kwargs)
        for r in response:
            decoded_response = tokenizer.decode(r.squeeze())
            decoded_response = decoded_response.replace("\\n", "\n")
            decoded_responses.append(decoded_response)
    
    return decoded_responses

def main():
    prompts = ['To My Fairy Fancies NAY,'] * 10 # loader.load_evaluation_prompts()[0]

    all_model_path = r'./logs'
    for model_path in os.listdir(all_model_path):
        print(model_path)
        if 'POEM' in model_path:
            model_log_dir = all_model_path + '/' + model_path 
            with open(model_log_dir + '/generate.txt', 'w') as f:
                response = generate(
                    model_path=model_log_dir + '/EPOCH400',
                    prompts=prompts
                    )
                    
                for generated_response in response:
                    f.write(f"=== GENERATED SEQUENCE ===\n")
                    f.write(generated_response)
                    f.write('\n')
            
if __name__ == '__main__':
    main()
import argparse
import os
import json
from tqdm import tqdm
# from PIL import Image
# import requests
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig, LlavaProcessor
from transformers import LlamaTokenizer, LlamaForCausalLM

import util

import sys
import torch
print(os.getcwd())

def llama_prompt(system_instruction, user_input):
    prompt = f"""<s>[INST] <<SYS>>
{system_instruction}
<</SYS>>

{user_input} [/INST]

"""
    return prompt

def format_prompt(class_name, concepts):
    instruction = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
    user = f"Answer yes/no/depends for whether the following concepts are salient for recognizing a '{class_name}': "  
    user += "; ".join(concepts) + ". "
    user += "Output format: <concept>: <answer>: <explanation>. Answer as a list."
    return instruction, user

def main(args):
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Model   
    model_path = f'/shared_data/p_vidalr/ryanckh/llama_hf/{args.model_name}'
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True, device_map='auto')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens(
        {
            
            "pad_token": "<PAD>",
        }
    )
    model.resize_token_embeddings(model.config.vocab_size + 1)
    model.eval()

    ## Prompt
    label_names = util.get_concepts(f"ip_omp/label_sets/{args.dataset_name}.txt")

    # Generate
    for label_name in tqdm(label_names[args.start_idx:args.end_idx]):
        if args.missing:
            concept_names = util.get_concepts(f"ip_omp/concept_sets_missing/{args.dataset_name}/{label_name}.txt")
            save_dir = f"ip_omp/outputs/{args.model_name}/{args.dataset_name}_missing"
        else:
            concept_names = util.get_concepts(f"ip_omp/concept_sets/{args.dataset_name}.txt")
            save_dir = f"ip_omp/outputs/{args.model_name}/{args.dataset_name}"
        
        for batch_i, concept_names_batch in enumerate(np.array_split(concept_names, args.batch_size)):
            prompt = llama_prompt(*format_prompt(label_name, concept_names_batch))
            inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation='longest_first').to(device)
            
            print("CLASS NAME: ", label_name)
            print(prompt)

            with torch.no_grad():
                generate_ids = model.generate(inputs['input_ids'], max_new_tokens=2048)

            out = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/{label_name}_batch{batch_i}.json", "w") as f:
                json.dump(out, f)
        
 
def prarseargs():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="llama-2-13b-chat")
    args.add_argument("--dataset_name", type=str, default="cifar10")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--start_idx", type=int, default=None)
    args.add_argument("--end_idx", type=int, default=None)
    args.add_argument("--missing", action='store_true', default=False)
    return args.parse_args()

if __name__ == '__main__':
    args = prarseargs()
    print(args)
    torch.multiprocessing.set_start_method('spawn')

    main(args)

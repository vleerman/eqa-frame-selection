import os
import re
import argparse
import json
import torch
import transformers
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed

set_seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_path = './results/open-eqa-llm_few_shot_llama3/'

with open('keys.txt', 'r') as f:
    for each in f.readlines():
        key, value = each.split('=')
        os.environ[key] = value.strip()

class llm_object_extractor:
    def __init__(self, model_id, prompt_path, nr_scenes, dataset):
        self.nr_scenes = nr_scenes
        self.dataset = dataset
        print(f"Loading prompt")
        self.prompt_template = self.load_prompt_template(prompt_path)
        print(f"Prompt loaded!")
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.eos = self.tokenizer("Q: ")["input_ids"]
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(self.DEVICE)
        self.model.eval()

    
    def load_prompt_template(self, path):
        prompt = ""

        with open(path, 'r') as f:
          for each in f.readlines():
            prompt += each
        return prompt

    
    def get_unique_episodes(self):
        self.questions = json.load(open("open-eqa/data/open-eqa-v0.json"))
        unique_history = set([question['episode_history'] for question in self.questions])
        done = set(x.split('.json')[0] for x in os.listdir(output_path) if x.endswith('json'))
        data = [x for x in unique_history if x.split('/')[1] not in done]

        episodes = []  
        
        if self.dataset != 'both':
            for each in data:
              if self.dataset in each:
                episodes.append(each)
        else:
            return sorted(data)
        return sorted(episodes)

    
    def get_model_result(self, method, verbose=False):
        episodes = self.get_unique_episodes()
        
        for each in tqdm(episodes[:self.nr_scenes]):
            temp_dict = {}
            scene_questions = [item for item in self.questions if item['episode_history'] == each]
            for q in scene_questions:
                input_text = self.prompt_template.format(question=q['question']).strip()
                input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.DEVICE)

                with torch.no_grad():
                    outputs = self.model.generate(**input_ids, max_new_tokens=16, temperature=0.2, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.eos)
                response = self.tokenizer.decode(outputs[0])

                if verbose:
                    print(response)
                
                objects = self.extract_list(response, 1)
                
                object_list = [item.strip(' []') for item in objects[0].split(',')]

                q['llm_objects'] = object_list

                # Fix this, so it is nicer to store per inference made
                q['method'] = method
                
            filename = each.split('/')[-1]
            with open(f"{output_path}{filename}.json", 'w') as f:
                json.dump(scene_questions, f)
                

    
    def extract_list(self, llm_output, nrOfLists):    
        return re.findall(r"\[[^\]]*\]", llm_output)[-nrOfLists:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="llm model to do inference")
    parser.add_argument("prompt_path", help="path to prompt template")
    parser.add_argument("--nr_scenes", default=5, help="specify number of scenes", type=int)
    parser.add_argument("--dataset", default='scannet-v0', help="specify dataset, can be either scannet-v0 or hm3d", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    method = args.model_id.split("/")[-1]
    
    llm = llm_object_extractor(args.model_id, args.prompt_path, args.nr_scenes, args.dataset)
    llm.get_model_result(method, args.verbose)
    print('Done')
    
    
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

output_path = './results/open-eqa-llm-objects/'

class llm_object_extractor:
    def __init__(self, model_id, prompt_path, nr_scenes, dataset):
        self.nr_scenes = nr_scenes
        self.dataset = dataset
        print(f"Loading prompt")
        self.prompt_template = self.load_prompt_template(prompt_path)
        print(f"Prompt loaded!")
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(self.DEVICE)

    
    def load_prompt_template(self, path):
        prompt = ""

        with open(path, 'r') as f:
          for each in f.readlines():
            prompt += each
        return prompt

    
    def get_unique_episodes(self):
        self.questions = json.load(open("open-eqa/data/open-eqa-v0.json"))
        unique_history = set([question['episode_history'] for question in self.questions])
        data = list(unique_history)
        episodes = []  
        
        if self.dataset != 'both':
            for each in data:
              if self.dataset in each:
                episodes.append(each)
        else:
            return sorted(data)
        return sorted(episodes)

    
    def get_model_result(self):
        episodes = self.get_unique_episodes()
        
        for each in tqdm(episodes[:self.nr_scenes]):
            temp_dict = {}
            scene_questions = [item for item in self.questions if item['episode_history'] == each]
            for q in scene_questions:
                input_text = self.prompt_template.format(question=q['question']).strip()
                input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.DEVICE)
            
                outputs = self.model.generate(**input_ids, max_new_tokens=64, temperature=0.2, do_sample=True)
                response = self.tokenizer.decode(outputs[0])

                objects = self.extract_list(response, 1)
                
                object_list = [item.strip(' []') for item in objects[0].split(',')]

                q['llm_objects'] = object_list
                
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
    args = parser.parse_args()
    
    llm = llm_object_extractor(args.model_id, args.prompt_path, args.nr_scenes, args.dataset)
    llm.get_model_result()
    print('Done')
    
    
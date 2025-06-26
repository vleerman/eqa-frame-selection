import torch
import json

import sys
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import matplotlib.pyplot as plt

from src.vlm import VLM

from torchvision import transforms

import pandas as pd

def main(cfg):
    img_height = cfg.img_height
    img_width = cfg.img_width
    num_frames = cfg.num_frames
    questions_path = cfg.questions_path
    img_dir = cfg.image_dir
    

    # load unique questions
    questions = json.load(open(questions_path))
    unique_hist = set([question['episode_history'] for question in questions])
    dataset = list(unique_hist)
    dataset = sorted([s.split('/')[1] for s in dataset if 'scannet' in s])
    
    # Load dataset
    df = pd.read_json(questions_path)
    vlm = VLM(cfg.vlm)

    for sceneNr in dataset:
        scene_questions = df[df['episode_history'].str.contains(sceneNr)]

        questions_data = scene_questions.to_dict('records')

        # Run all questions
        cnt_data = 0
        results_all = []
        for question_ind in tqdm(range(len(questions_data))):

            # Extract question
            question_data = questions_data[question_ind]
            question = question_data["question"]
            answer = question_data["answer"]
            scene = question_data["episode_history"]
            question_data["method"] = "euc"

            # delete empty entry
            if type(question_data["extra_answers"]) == float:
                del question_data["extra_answers"]
            print(f"\n========\nIndex: {question_ind} Scene: {scene}")


            print(f"Question:\n{question}\nAnswer: {answer}")

            # Set data dir for this question - set initial data to be saved
            result = {"question_ind": question_ind}

            # Run steps
            for cnt_step in range(num_frames):
                step_name = f"step_{cnt_step}"
                result[step_name] = {}
                # Load image
                if cnt_step < 10:
                    rgb = Image.open(f"{img_dir}/{sceneNr}/00000{cnt_step}.png").convert("RGB")
                else:
                    rgb = Image.open(f"{img_dir}/{sceneNr}/0000{cnt_step}.png").convert("RGB")

                # Get VLM relevancy
                prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with Yes or No."
                smx_vlm_rel = vlm.get_loss(rgb, prompt_rel, ["Yes", "No"])
                
                # Save data
                result[step_name]["smx_vlm_rel"] = smx_vlm_rel

            relevancy_all = []
            for step in range(num_frames):
                smx_vlm_rel = result[f"step_{step}"]["smx_vlm_rel"]                     # Get score of current image
                relevancy_all.append(smx_vlm_rel[0])                                    # Append 'Yes' score
            
            relevancy_ord = np.flip(np.argsort(relevancy_all))

            # Episode summary
            print(
                f"Top 3 steps with highest relevancy with value: {relevancy_ord[:3]} {[relevancy_all[i] for i in relevancy_ord[:3]]}"
            )
            # Save data
            rel_sort = [relevancy_all[i] for i in relevancy_ord]
            indx_prob = dict(zip(relevancy_ord.tolist(), rel_sort))
            question_data["euc_ranked_frames"] = [(int(k),v) for k,v in indx_prob.items()]
            cnt_data += 1
            
        name = questions_data[0]["episode_history"].split('/')[-1]
        
        with open(os.path.join(cfg.output_dir, f"{name}.json"), "w") as f:
            json.dump(questions_data, f)
            
        print(f"\n== Summary")
        print(f"Number of data collected: {cnt_data}")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    args = argparse.Namespace()
    args.cfg_file = "vlm_config.yaml"
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    

    # run
    print(f"***** Running {cfg.exp_name} *****")
    main(cfg)
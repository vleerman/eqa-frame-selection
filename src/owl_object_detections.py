import os
import json
import copy
import torch
import pickle
from tqdm import tqdm
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get json from scannet with objects to look for
def get_questions_from_directory(object_path, output_path, overwrite=False):
    skip = []
    if not overwrite:
        skip = [x for x in os.listdir(output_path) if x.endswith('.json')]
        
    questions_per_scene = []
    for each in os.listdir(object_path):
        if each.endswith('.json') and ("scannet" in each) and (each not in skip):
            with open(object_path + each, 'r') as f:
                questions_per_scene.append(json.load(f))

    return questions_per_scene


# Load all objects for every question in a scene
def get_objects_per_scene(scene_json, key):
    objects = []
    for each in scene_json:
        objects.append(each[key])
    return objects


# Load all images from an episode
def load_full_episode(epi_path):
    all_images = []
    for each in sorted(os.listdir(epi_path)):
        # load image
        if each.endswith('.png'):
            all_images.append(Image.open(os.path.join(epi_path, each)).convert('RGB'))
    width, height = all_images[0].size
    return all_images, width, height


# Object detection on all images
'''
Takes as input a set of images, list of text_queries, model checkpoint and batch_size
'''
def object_detection(imgs, text_queries, threshold, batch_size=4):
    all_results = []

    for i in range(0, len(imgs), batch_size):

        batch_imgs = imgs[i:i+batch_size]

        target_sizes = torch.tensor([[max(img.size), max(img.size)] for img in batch_imgs]).to(DEVICE)

        inputs = processor(text=text_queries*batch_size, images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)
            all_results.extend(results)

    return all_results


def get_frame_scores(results):
    od_results = {}
    for i, each in enumerate(results):
        best_score = {}
        for score, label in zip(each['scores'], each['labels']):
            if best_score:
                label_seen = False
                for x, y in best_score.items():
                    if x == label.item():
                        label_seen = True
                        best_score[x] = max(y, score.item())
                if label_seen:
                    continue
            best_score[label.item()] = score.item()
        od_results[i] = best_score
    return od_results 


def get_sorted_frames(results, nrOfObjects):
    frames = []

    if nrOfObjects == 0:
        return [(0,0,'no_objects_to_look_for')]
    
    for j in range(nrOfObjects, 0, -1):
        for i in results:
            if len(results[i]) == j:
                score = sum(results[i].values())/j
                #print(result)
                frames.append((i, j, score))
    frames = sorted(frames, key=lambda x: (x[1], x[2]), reverse=True)
    
    if len(frames) < 30:
        for i in results:
            if len(results[i]) == 0:
                frames.append((i, 0, 'no_object_found'))
    return frames

# Project results onto images
def project_results(results, imgs, text_queries):
    img_copy = copy.deepcopy(imgs)

    for i in range(len(imgs)):
        draw = ImageDraw.Draw(img_copy[i])
    
        scores = results[i]["scores"].tolist()
        labels = results[i]["labels"].tolist()
        boxes = results[i]["boxes"].tolist()
    
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

    return img_copy

# Save images to path if needed
def save_photos(imgs, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    for i in range(len(imgs)):
        imgs[i].save(f"{path}/{i}.png")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    args = argparse.Namespace()
    args.cfg_file = "src/local_config.yaml"
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    set_seed(cfg.seed)
    
    # Get questions with object to look for
    questions = get_questions_from_directory(object_path=cfg.object_path, output_path=cfg.output_path, overwrite=True)[:cfg.nr_scenes] 

    # Load models, Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_id).to(DEVICE)
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # For each scene, extract all objects and get images, detect and sort detections and save json
    # if verbose, save the scene_detections and images with bounding boxes
    for scene in tqdm(questions):
        objects = get_objects_per_scene(scene, cfg.object_type)
        scene_frame_path = scene[0]['episode_history'].split('/')[-1]
        imgs, w, h = load_full_episode(cfg.image_path + scene_frame_path)
        scene_detections = []
        for i, obj in tqdm(enumerate(objects), leave=False):
            if len(obj) == 0:
                scene[i][cfg.object_type + '_frames'] = 'No_Objects_Extracted'
                continue
            result = object_detection(imgs, obj, cfg.threshold)
            scene_detections.append(result)
            frame_scores = get_frame_scores(result)
            sorted_frames = get_sorted_frames(frame_scores, len(obj))
            scene[i][cfg.object_type + '_frames'] = sorted_frames
            if cfg.verbose:
                processed_imgs = project_results(result, imgs, obj)
                output_verbose_path = f"{cfg.verbose_path}/{scene_frame_path}/question{i}"
                save_photos(processed_imgs, output_verbose_path)
                output_verbose_path += ".pkl"
                with open(output_verbose_path, 'wb') as f:
                    pickle.dump(result, f)

        output_directory = cfg.output_path + scene_frame_path + '.json'
        json.dump(scene, open(output_directory, "w"))
      
    print('Done')
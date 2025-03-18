import os
import json
from PIL import Image, ImageDraw

CHECKPOINT = "google/owlv2-base-patch16-ensemble"
SOURCE_IMAGE_PATH = f"../eqa-test/scannet/frames/"

# Get json from scannet/hm3d with objects to look for
def get_questions_from_directory(dataset = ['scannet', 'hm3d'], object_path = './results/open-eqa-llm-objects/'):
    questions_per_scene = []
    if len(dataset) == 2:
        for each in os.listdir(object_path):
            if each.endswith('.json'):
                print(each)
    else:
        for each in os.listdir(object_path):
            if each.endswith('.json') and dataset[0] in each:
                with open(object_path + each, 'r') as f:
                    questions_per_scene.append(json.load(f))

    return questions_per_scene


# Load all objects for every question in a scene
def get_objects_per_scene(scene_json):
    objects = []
    for each in scene_json:
        objects.append(each['llm_objects'])
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
def object_detection(imgs, text_queries, checkpoint, batch_size=4):
    all_results = []
    #target_sizes = torch.Tensor([[max(h,w), max(h,w)]])

    # PLace somewhere else
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(DEVICE)
    processor = AutoProcessor.from_pretrained(checkpoint)

    # Implemented batches
    for i in range(0, len(imgs), batch_size):

        batch_imgs = imgs[i:i+batch_size]

        target_sizes = torch.tensor([[max(img.size), max(img.size)] for img in batch_imgs]).to(DEVICE)

        inputs = processor(text=text_queries*batch_size, images=batch_imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            results = processor.post_process_object_detection(outputs, threshold=THRESHOLD, target_sizes=target_sizes)
            all_results.extend(results)

    return all_results


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="llm model to do inference")
    parser.add_argument("prompt_path", help="path to prompt template")
    parser.add_argument("--nr_scenes", default=5, help="specify number of scenes", type=int)
    parser.add_argument("--dataset", default='scannet-v0', help="specify dataset, can be either scannet-v0 or hm3d", type=str)
    args = parser.parse_args()
    
    llm = llm_object_extractor(args.model_id, args.prompt_path, args.nr_scenes, args.dataset)
    llm.get_model_result()
    print('Done')
    '''
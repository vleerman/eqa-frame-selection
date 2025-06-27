import os
import argparse
import json
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class nlp_object_extractor:
    def __init__(self, dataset, questions_path, output_path):
        self.nlp = spacy.load("en_core_web_sm")
        self.dataset = dataset
        self.question_path = questions_path
        self.output_path = output_path
    
    def get_unique_episodes(self):
        self.questions = json.load(open(self.question_path))
        unique_history = set([question['episode_history'] for question in self.questions])
        data = list(unique_history)
        episodes = []  
        
        for each in data:
            if self.dataset in each:
                episodes.append(each)

        return sorted(episodes)


    def get_objects(self, question, verbose=False):
        docs = []
        ban_list = ['color', 'food', 'space', 'brand']
        
        doc = self.nlp(question)
            
        objects = []
        for token in doc:
            if verbose:
                print(token.text, token.dep_, token.head, token.pos_)
                
            tokenlowered = token.text.lower()
            if tokenlowered in stopwords.words('english') or tokenlowered in ban_list:
                continue
            if any(tokenlowered in x for x in objects):         # Check if token already there
                continue
            if token.dep_ == 'amod':                              # compound relation between words
                if any(token.head.lemma_ in x for x in objects):         # Check if token
                    continue
                objects.append(f"{token.text.lower()} {token.head.lemma_}")
            if token.dep_ == 'compound':                          # compound relation between words
                if any(token.head.lemma_ in x for x in objects):
                    continue
                objects.append(f"{token.text.lower()} {token.head.lemma_}")
            if token.dep_ == 'pobj' or token.dep_ == 'dobj' or (token.dep_ == 'conj' and token.pos_ == 'NOUN'):                                  # direct object
                objects.append(token.lemma_)
            if token.dep_ == 'nsubj':                                 # subject
                word = token.lemma_
                if any(word in x for x in objects):
                    continue
                objects.append(word)
            if token.dep_ == "attr" and token.pos_ == 'NOUN':
                if any(token.lemma_ in x for x in objects):
                    continue
                objects.append(token.lemma_)
        return objects
    
    def get_nlp_result(self):
        episodes = self.get_unique_episodes()
        
        for each in episodes:
            scene_questions = [item for item in self.questions if item['episode_history'] == each]
            for q in scene_questions:
                objects = self.get_objects(q['question'])
                
                q['nlp_objects'] = objects
                q['method'] = 'nlp'
                
            filename = each.split('/')[-1]
            with open(f"{self.output_path}{filename}.json", 'w') as f:
                json.dump(scene_questions, f)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_path", default="open-eqa/data/open-eqa-v0.json", help="specify path to questions", type=str)
    parser.add_argument("--output_path", default='./results/open-eqa-nlp-objects/', help="specify output_path", type=str)
    args = parser.parse_args()
    
    obj_extractor = nlp_object_extractor('scannet-v0', args.question_path, args.output_path)
    obj_extractor.get_nlp_result()
    print('Done')

    

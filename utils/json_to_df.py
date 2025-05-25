# https://lyaaaan.medium.com/huggingface-datasets-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC-%EB%8B%A4%EB%A3%A8%EA%B8%B0-7acbe0458947 참고
import datasets
import json
import pandas as pd
from datasets import Dataset
from glob import glob
from tqdm import tqdm

def json_convert_dataframe(json_paths, save_path):
    en_list = []
    ko_list = []

    for i in tqdm(range(len(json_paths)), ncols = 70):
        json_path = json_paths[i]
        
        if json_path.split("/")[-6] == 'Broadcast_content_Korean_English_translation_corpus':
            with open(json_path, 'r') as f:
                translation_json = json.load(f)
            
            ko = translation_json['원문']
            en = translation_json['최종번역문']

            ko_list.append(ko)
            en_list.append(en)
        
        elif json_path.split("/")[-3] == 'Korean_English_translation_corpus_social science':
            with open(json_path, 'r') as f:
                translation_json = json.load(f)
                translation_json = translation_json['data']
            
            for d in translation_json:
                ko_list.append(d['ko'])
                en_list.append(d['en'])
        
        elif json_path.split("/")[-3] =='Korean_English_translation_corpus_technological_science':
            with open(json_path, 'r') as f:
                translation_json = json.load(f)
                translation_json = translation_json['data']
            
            for d in translation_json:
                ko_list.append(d['ko'])
                en_list.append(d['en'])
        
        elif json_path.split("/")[-5] =='Korean_English_translation_parallel_corpus_data_in_technical_science_field':
            with open(json_path, 'r') as f:
                translation_json = json.load(f)
                translation_json = translation_json['data']
            
            for d in translation_json:
                ko_list.append(d['ko'])
                en_list.append(d['en'])
        
        elif json_path.split("/")[-5] =='Korean_multilingual_translation_corpus_basic_science':
            with open(json_path, 'r') as f:
                translation_json = json.load(f)
                translation_json = translation_json['paragraph']
            
            for d in translation_json:
                sentences = d['sentences']
                for s in sentences:
                    ko_list.append(s['src_sentence'])
                    en_list.append(s['tgt_sentence'])
        
        elif json_path.split("/")[-3] =='Specialty_Korean_English_Corpus':
            with open(json_path, 'r') as f:
                translation_json = json.load(f)
            
            for d in translation_json:
                ko_list.append(d['한국어'])
                en_list.append(d['영어'])

    df = pd.DataFrame({
        "ko" : ko_list,
        "en": en_list
    })

    df.to_csv(save_path, index = False)
    
if __name__ == '__main__':
    
    json_paths = glob('/workspace/paper_translation/paper_translation_data/Broadcast_content_Korean_English_translation_corpus/Official_open_data/Training/labeling/Arts/*.json') +\
                glob("/workspace/paper_translation/paper_translation_data/Korean_English_translation_corpus_social science/*/*.json") +\
                glob("/workspace/paper_translation/paper_translation_data/Korean_English_translation_corpus_technological_science/*/*.json") +\
                glob("/workspace/paper_translation/paper_translation_data/Korean_English_translation_parallel_corpus_data_in_technical_science_field/data/*/labeling/*.json") +\
                glob("/workspace/paper_translation/paper_translation_data/Korean_multilingual_translation_corpus_basic_science/Official_opendata/*/labeling/*.json") +\
                glob("/workspace/paper_translation/paper_translation_data/Specialty_Korean_English_Corpus/*/*.json")
    
    save_path = "/workspace/paper_translation/paper_translation_data/translation_dataset.csv"
    
    json_convert_dataframe(json_paths, save_path)



        


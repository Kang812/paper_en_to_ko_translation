import pandas as pd
from sklearn.model_selection import train_test_split
import datasets
from datasets import Dataset


train = pd.read_csv("/workspace/paper_translation/paper_translation_data/train.csv")
valid = pd.read_csv("/workspace/paper_translation/paper_translation_data/valid.csv")
test = pd.read_csv("/workspace/paper_translation/paper_translation_data/test.csv")

def dataframe_to_huggingface_dataset(train_df_path,
                                     valid_df_path,
                                     test_df_path,
                                     save_disk_path):
    
    train = pd.read_csv(train_df_path)
    valid = pd.read_csv(valid_df_path)
    test = pd.read_csv(test_df_path)

    for index, df in enumerate([train, valid, test]):
        df_ko = df['ko'].to_list()
        df_en = df['en'].to_list()

        if index == 0:
            json_data = []
            for ko, en in zip(df_ko, df_en):
                a = {'ko':ko, 'en':en}
                json_data.append(a)
            train_dataset = Dataset.from_list(json_data)

        elif index == 1:
            json_data = []
            for ko, en in zip(df_ko, df_en):
                a = {'ko':ko, 'en':en}
                json_data.append(a)
            valid_dataset = Dataset.from_list(json_data)
        
        elif index == 2:
            json_data = []
            for ko, en in zip(df_ko, df_en):
                a = {'ko':ko, 'en':en}
                json_data.append(a)
            test_dataset = Dataset.from_list(json_data)

    
    class_dataset = datasets.DatasetDict({'train' : train_dataset,
                        'valid': valid_dataset,
                        'test' : test_dataset})  
    
    class_dataset.save_to_disk(save_disk_path)

if __name__ == '__main__':
    train_df_path = "/workspace/paper_translation/paper_translation_data/train.csv"
    valid_df_path = "/workspace/paper_translation/paper_translation_data/valid.csv"
    test_df_path = "/workspace/paper_translation/paper_translation_data/test.csv"

    save_disk_path = '/workspace/paper_translation/paper_translation_data/translation_dataset/'

    dataframe_to_huggingface_dataset(train_df_path, valid_df_path, test_df_path, save_disk_path)

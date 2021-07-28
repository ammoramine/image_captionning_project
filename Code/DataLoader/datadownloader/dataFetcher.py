"""
    module to fetch information from a json file, connecting ids of images to captions,
    parse the informations and write it on excel file
"""
from torch_snippets import *
import json,os
import numpy as np
from openimages.download import _download_images_by_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dirFile = os.path.dirname(__file__)
path_json = os.path.join(dirFile, '../../Data/open_images_train_v6_captions.jsonl')

#TODO get the data from

def fetch_info_from_json(path_json,N=10000):
    """Loop through the content of the JSON file and fetch the information of the first 100,000 images
        inputs:
        path_json : contains information about id of image caption, and id annotator
        N : numer of image to download
    """
    with open(path_json, 'r') as json_file:
        json_list = json_file.read().split('\n')
        np.random.shuffle(json_list)
        data = []
        for ix, json_str in Tqdm(enumerate(json_list), N):
            if ix == N: break
            try:
                result = json.loads(json_str)
                x = pd.DataFrame.from_dict(result, orient='index').T
                data.append(x)
            except:
                pass
        return data

def split_data_to_training_and_validation_dataset(data):
    np.random.seed(10)
    data = pd.concat(data)
    data['train'] = np.random.choice([True,False], size=len(data),p=[0.95,0.05])
    data.to_csv('data.csv', index=False)
    return data

if __name__ == '__main__':
    defined_variables = globals().keys()
    if "data" not in defined_variables:
        data = fetch_info_from_json(path_json,10000)
        data = split_data_to_training_and_validation_dataset(data)
        dirFile = os.path.dirname(__file__)
        data.to_excel(os.path.join(dirFile, "../../Data/dataInfo.xlsx"))
    else:
        print("data is already defined")

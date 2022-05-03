import json
import numpy as np
from utils import make_request
import os


DATASET_URL = "https://us-central1-face-images-api.cloudfunctions.net/images"
URLS_PATH = "urls.json"


def load_dataset():

    if not URLS_PATH in os.listdir():

        data = make_request(DATASET_URL)['urls'] 
        with open(URLS_PATH, 'w') as f:
            json.dump(data, f)

    else:

        data = json.load(open(URLS_PATH, 'r'))

    return data    
            

class Dataset:

    def __init__(self) -> None:
        self.data = load_dataset()
        self.__preprocess()
    
    def shuffle(self):
        np.random.shuffle(self.data)

    def by(self, /, age=None, gender=None):

        age = age if str(age) != 'None' else None
        gender = 0 if gender=='male' else 1 if str(gender) != 'None' else None

        try:
           return self.__get_item(age, gender) 

        except Exception as e:
            i  = np.random.randint(0, len(self.__options))
            key = self.__option_keys[i]
            return  np.random.choice(self.__options[key])  


    def __get_item(self, age, gender):

        if age is None and gender is None:
            i  = np.random.randint(0, len(self.__options))
        elif age is None:
            i = np.random.choice([i for i, k in enumerate(self.__option_keys) if k[1]==gender])
        elif gender is None:
            i = np.random.choice([i for i, k in enumerate(self.__option_keys) if k[0]==age])
        else: 
            i = np.random.choice([i for i, k in enumerate(self.__option_keys) if k==(age, gender)])

        key = self.__option_keys[i]
        return  np.random.choice(self.__options[key])     
               
    def __preprocess(self):

        self.__options = dict()
        for item in self.data:

            try:

                age, gender, *_ = item['name'].split("_") 
                age = int(age)
                gender = int(gender)

            except: 

                continue    

            if not (age, gender) in self.__options:

               self.__options[(age, gender)] = []

            self.__options[(age, gender)].append(item['url'])

        self.__option_keys = list(self.__options)
            
dataset = Dataset()                       
import numpy as np
import os


datasets = {
  "UTKFace":  "G:/more_datasets/UTKFace.tar/UTKFace/UTKFace",
  "UTK_part1":  "G:/more_datasets/part1.tar/part1/part1",
  "UTK_part2":  "G:/more_datasets/part2.tar/part2/part2",
  "UTK_part3":  "G:/more_datasets/part3.tar/part3/part3",
#   "wiki_crop":  ["G:/more_datasets/wiki_crop/wiki_crop"]
}

def is_image(file):
    return file.split(".")[-1].lower() in ['jpg', "png", "jpeg"]


class Dataset:
    def __init__(self, data) -> None:
        self.data = data 
        self.i = -1  
        self.__preprocess()

    def __call__(self):
        try:
            self.i+=1
            return self.data[self.i]
        except:
            self.i = 0
            return self() 
    
    def shuffle(self):
        np.random.shuffle(self.data)

    # def by(self, /, age:UnionType[int, None]=None, gender:UnionType[str, None]=None):
    def by(self, /, age=None, gender=None):
        age = age if str(age) != 'None' else None
        gender = 0 if gender=='male' else 1 if str(gender) != 'None' else None

        print(age, gender)

        try:
           return self.__get_item(age, gender) 
        except Exception as e:
            print(e)
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
        for p in self.data:
            try:
                age, gender, *_ = os.path.split(p)[-1].split("_") 
                age = int(age)
                gender = int(gender)
            except: 
                continue    
            if not (age, gender) in self.__options:
               self.__options[(age, gender)] = []
            self.__options[(age, gender)].append(p)
        self.__option_keys = list(self.__options)
            
        
        



def load_dataset(name):
    dataset_path = datasets[name]
    results = []
    if isinstance(dataset_path, str):
        for p in os.listdir(dataset_path):
           
            try:
                if not is_image(p): continue
                fp = os.path.join(dataset_path, p)
                results.append(fp)
                # yield fp
            except: 
                continue
    
    elif isinstance(dataset_path, list):
        dataset_path = dataset_path[0]
        for p in os.listdir(dataset_path):
            d = os.path.join(dataset_path, p)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                try: 
                    if not is_image(fp): continue
                    results.append(fp)
                    # yield fp
                except: continue
    np.random.shuffle(results)   
  
    return Dataset(results)

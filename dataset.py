import os
from torch.utils.data import Dataset
from util import unicodeToAscii
import random

class NamesDataset(Dataset):
    def __init__(self, root_dir, all_letters, split_ratio=0.8):
        self.root_dir = root_dir
        self.split_ratio = split_ratio
        self.languages = [lang.replace(".txt", "") for lang in os.listdir(root_dir) if lang.endswith('.txt')]
        self.name_list = []

        for lang in self.languages:
            lang_dir = os.path.join(root_dir, lang + ".txt")
            lines = open(lang_dir, encoding='utf-8').read().strip().split('\n')
            names = [unicodeToAscii(line, all_letters) for line in lines]
            self.name_list.extend([(name, lang) for name in names])

        random.shuffle(self.name_list)
        self.train_data = self.name_list[:int(self.split_ratio*len(self.name_list))]
        self.test_data = self.name_list[int(self.split_ratio*len(self.name_list)):]

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        return self.train_data[index]
    
    def get_test_data(self):
        return self.test_data
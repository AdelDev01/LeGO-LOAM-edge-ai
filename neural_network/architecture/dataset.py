import torch
from torch.utils.data import Dataset
import glob
import numpy as np

class Range_Ground_Dataset(Dataset):

    def __init__(self, folder_path):
        super().__init__()

        self.range_list = sorted(glob.glob(folder_path + "/range_*.npy"))
        self.ground_list = sorted(glob.glob(folder_path + "/ground_*.npy"))

        for i,j in zip(self.range_list,self.ground_list):
            if i.split("_")[1] != j.split("_")[1] or len(self.range_list) != len(self.ground_list):
                raise ValueError("I dati non combaciano")


    def __len__(self):
        
        return len(self.range_list)

    def __getitem__(self, index):

        range_img = np.load(self.range_list[index])
        ground_img = np.load(self.ground_list[index])

        range_img = range_img.copy()
        range_img[range_img >= 1e10] = 0.0
        range_img = range_img / 100.0

        return torch.from_numpy(range_img).float().unsqueeze(0), torch.from_numpy(ground_img).float().unsqueeze(0)

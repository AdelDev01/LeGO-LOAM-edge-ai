import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import Range_Ground_Dataset

BASE = "/home/adel/tesi/dataset_export/run_"

train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

val_ids = [28, 29, 30, 31]

test_ids = [32, 33, 34, 35, 36, 37, 38, 39]

training_dataset = [BASE + str(i) for i in train_ids]
validation_dataset = [BASE + str(i) for i in val_ids]
test_dataset = [BASE + str(i) for i in test_ids]


training_run_list = []
validation_run_list = []
test_run_list = []

for i in training_dataset:
    training_run_list.append(Range_Ground_Dataset(i))

for j in validation_dataset:
    validation_run_list.append(Range_Ground_Dataset(j))

for k in test_dataset:
    test_run_list.append(Range_Ground_Dataset(k))



training_combined_dataset = ConcatDataset(training_run_list)
validation_combined_dataset = ConcatDataset(validation_run_list)
test_combined_dataset = ConcatDataset(test_run_list)

training_loader = DataLoader(training_combined_dataset, batch_size= 16, shuffle=True)
validation_loader = DataLoader(validation_combined_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(test_combined_dataset, batch_size= 16, shuffle=False)
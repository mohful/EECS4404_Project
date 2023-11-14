import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

'''
The purpose of this file is to visualize what the dataset looks like.
In classifier.py, i didnt know how to plot the training and validation dataset split, so i made a whole new file
and checked if I could visualize the data and see if its balanced or not 
(very crucial thing to know so you can explain yourself if the model starts acting up)
I dont think we have to submit this file, so please ignore it and only use it for visualization of data
'''

np.random.seed(0)
torch.manual_seed(0)

sns.set_style('darkgrid')

transformations = {
    "train": transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
}

train_dataset_folder = "dataset/train"
test_dataset_folder = "dataset/test"

dataset = datasets.ImageFolder(root=train_dataset_folder, transform=transformations["train"])
idx2class = {v: k for k, v in dataset.class_to_idx.items()}

def distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict

def dataset_bar_chart(dict_obj, plot_title, **kwargs):
    return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)

plt.figure(figsize=(15,8))
dataset_bar_chart(distribution(dataset), plot_title="Entire Dataset (before train/val/test split)")
plt.show()

indices = list(range(len(dataset)))

np.random.shuffle(indices)

val_split_index = int(np.floor(0.2 * len(dataset)))

training_index, validation_index = indices[val_split_index:], indices[:val_split_index]

training = SubsetRandomSampler(training_index)
validation = SubsetRandomSampler(validation_index)

test_dataset = datasets.ImageFolder(root=test_dataset_folder, transform=transformations["test"])

train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=100, sampler=training_index)
val_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=100, sampler=validation_index)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=100)

def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:    
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else: 
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
dataset_bar_chart(get_class_distribution_loaders(train_loader, dataset), plot_title="Train Set", ax=axes[0])
dataset_bar_chart(get_class_distribution_loaders(val_loader, dataset), plot_title="Val Set", ax=axes[1])

plt.show()

single_batch = next(iter(train_loader))
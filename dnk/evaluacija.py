import model_dna
import time
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
# deep learning/vision libraries
from torch.utils.data import Dataset, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, models, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# numeric and plotting libraries
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 18

data_dir = 'Data/Data6000'
path1 = 'Model/state_dict_model.pt'
custom_cnn = model_dna.CustomCNN().to(device)
optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, custom_cnn.parameters()))
criterion = nn.CrossEntropyLoss()

assert os.path.exists(data_dir), f'Could not find {data_dir} in working directory {os.getcwd()}n'
dirs_exist = [os.path.exists(os.path.join(data_dir, x)) for x in ['train', 'val']]
assert all(dirs_exist), f'Could not find train/val dirs check if you have train and val directly under {data_dir}.'
data_numpy_list = {}
data_numpy_list['train'] = [x for x in glob.glob(os.path.join(data_dir, 'Train/**/*.npy'), recursive=True)]
data_numpy_list['val'] = [x for x in glob.glob(os.path.join(data_dir, 'Val/**/*.npy'), recursive=True)]
data_numpy_list['test'] = [x for x in glob.glob(os.path.join(data_dir, 'Test/**/*.npy'), recursive=True)]
    #print(data_numpy_list)

tensordat = {}
for x in ['train', 'val', 'test']:
    data_list = []
    label_list = []
    for ind in range(len(data_numpy_list[x])):
        data_slice_file_name = data_numpy_list[x][ind]
        data_i = np.load(data_slice_file_name)
        # print(data_i.shape)
        idx = data_slice_file_name.rfind("\\")
        data_list.extend(data_i)
        num = [int(data_slice_file_name[idx - 1])-1] * len(data_i)
        label_list.extend(num)
    data_list = torch.tensor(data_list)
    label_list = torch.tensor(label_list)
    tensordat[x] = TensorDataset(data_list, label_list)

dataloaders = {x: torch.utils.data.DataLoader(tensordat[x], batch_size=100, shuffle=True) for x in ['train', 'val', 'test']}

a = model_dna.model_eval(custom_cnn,criterion, optimizer_conv, path1, dataloaders)



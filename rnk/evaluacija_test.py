import model_shared
import time
import os
import pretprocesiranje
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
# deep learning/vision libraries

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

data_dir = 'Data/'
path1 = 'Model/state_dict_model.pt'
custom_cnn = model_shared.CustomCNN().to(device)
optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, custom_cnn.parameters()))
criterion = nn.CrossEntropyLoss()

podaci_za_test = [x for x in glob.glob(os.path.join(data_dir, 'val/**/*.npy'), recursive=True)]
datasets1 = {x: model_shared.NumpyDataset(podaci_za_test, model_shared.data_transforms) for x in ['val']}
podaci = {x: torch.utils.data.DataLoader(datasets1[x], batch_size=BATCH_SIZE, shuffle=True) for x in
                   ['val']}

model_shared.model_eval(custom_cnn,criterion, optimizer_conv, path1, podaci)



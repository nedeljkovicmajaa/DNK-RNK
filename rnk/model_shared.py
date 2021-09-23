import time
import os
import pretprocesiranje
import glob
import math
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

# hyperparameters
LR = 0.01  # learning rate
TRAINING_ITER = 3000  # iteration times
BATCH_SIZE = 18  # batch size of input

SEQUENCE_LENGTH = 164  # sequence length of input
EMBEDDING_SIZE = 4  # char embedding size(sequence width of input)

# CONV_SIZE = 3    #first filter size
# CONV_DEEP = 128   #number of first filter(convolution deepth)

stride_sizes = [1, 1, 1, 1]  # the strid in each of four dimensions during convolution
kernel_sizes = [1, 1, 164, 1]  # pooling window size

FC_SIZE = 128  # nodes of full-connection layer
NUM_CLASSES = 2  # classification number

DROPOUT_KEEP_PROB = 0.5  # keep probability of dropout

# These will be executed over every training/val image (handled by PyTorch's dataloader)

class NumpyDataset(Dataset):

    def __init__(self, data_numpy_list, data_transforms):
        self.data_numpy_list = data_numpy_list
        self.transform = data_transforms
        self.data_list = []
        self.label_list = []
        for ind in range(len(self.data_numpy_list)):
            data_slice_file_name = self.data_numpy_list[ind]
            data_i = np.load(data_slice_file_name)
            idx = data_slice_file_name.rfind("\\")
            self.data_list.append(data_i)
            self.label_list.append(int(data_slice_file_name[idx-1]))

    def __getitem__(self, index):

        data = np.asarray(self.data_list[index])
        label = np.asarray(self.label_list[index])
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        return data, label

    def __len__(self):
        return len(self.data_numpy_list)

data_transforms = {
    transforms.Compose([
        transforms.ToTensor(),
    ])
}

# define placeholder
class CustomCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, (3, 1), stride=1)
        self.conv2 = nn.Conv2d(1, 32, (4, 1), stride=1)
        self.conv3 = nn.Conv2d(1, 32, (5, 1), stride=1)
        self.conv4 = nn.Conv2d(1, 32, (6, 1), stride=1)
        self.maxpool1 = nn.MaxPool2d((159, 1))
        self.maxpool2 = nn.MaxPool2d((159, 2))
        self.maxpool3 = nn.MaxPool2d((159, 3))
        self.maxpool4 = nn.MaxPool2d((159, 4))
        self.fc1 = nn.Linear(512,128)
        self.dropout = nn.Dropout(DROPOUT_KEEP_PROB)
        self.fc2 = nn.Linear(128,2)
        self.sf = nn.Softmax(1)

    def forward(self, x):
        y1 = self.conv4(x)
        y1 = self.relu(y1)
        y2 = self.conv4(x)
        y2 = self.relu(y2)
        y3 = self.conv4(x)
        y3 = self.relu(y3)
        y4 = self.conv4(x)
        y4 = self.relu(y4)
        y = torch.cat([y1, y2, y3, y4], -3)
        y = self.maxpool1(y)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.sf(y)
        return y


def train_model(model, criterion, optimizer, num_epochs, path):
    start_time = time.time()

    metrics = defaultdict(list)
    total = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            best_acc = 0
            # Iterate over data.

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                for i in range(len(preds)):
                    if(preds[i] == labels.data[i]):
                        if(preds[i] == 1):
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if (preds[i] == 1):
                            fp += 1
                        else:
                            fn += 1

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

                total += labels.size(0)
                correct += (preds == labels.data).sum()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = 100 * float(correct) / total

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

            metrics[phase + "_loss"].append(epoch_loss)
            metrics[phase + "_acc"].append(epoch_acc)


    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc: .4f}')

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f'Sens: {sensitivity: .4f} Spec: {specificity: .4f} F1: {f1: .4f}')

    torch.save(model.state_dict(), path)
    # load best model weights
    return model, metrics

def model_eval(model, criterion ,optimizer, PATH, podaci):
    model.load_state_dict(torch.load(PATH))
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    correct = 0
    tp=0
    tn=0
    fp=0
    fn=0
    phase = 'test'
    for inputs, labels in podaci[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels.long())
        for i in range(len(preds)):
            if (preds[i] == labels.data[i]):
                if (preds[i] == 1):
                    tp += 1
                else:
                    tn += 1
            else:
                if (preds[i] == 1):
                    fp += 1
                else:
                    fn += 1

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

        total += labels.size(0)
        correct += (preds == labels.data).sum()

    epoch_loss = running_loss / 201
    epoch_acc = 100 * float(correct) / total
    mcc = (tn * tp-fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) 
    print(f'MCC: {mcc: .4f}')
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    putanja1 = "Data/miRBase_set.csv"
    putanja2 = "Data/putative_mirtrons_set.csv"
    podaci = pretprocesiranje.ucitavanje_podataka(putanja1, putanja2)

    vektorizovani_podaci = pretprocesiranje.vektorizacija(podaci)
    X_train, y_train, X_test, y_test = pretprocesiranje.podela_podataka(vektorizovani_podaci)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test2 = X_test[:200]
    y_test2 = y_test[:200]

    X_test1 = X_test[200:]
    y_test1 = y_test[200:]

    dataset_sizes = {'train': len(X_train),
                    'val': len(X_test1),
                     'test': len(X_test2)}

    dataset = {
        'train': [X_train, y_train],
        'val': [X_test, y_test]
    }
    data_dir = 'Data/'
    path1 = 'Model/state_dict_model.pt'

    # basic error checking to check whether you correctly unzipped the dataset into the working directory
    assert os.path.exists(data_dir), f'Could not find {data_dir} in working directory {os.getcwd()}n'
    dirs_exist = [os.path.exists(os.path.join(data_dir, x)) for x in ['train', 'val']]
    assert all(dirs_exist), f'Could not find train/val dirs check if you have train and val directly under {data_dir}.'
    data_numpy_list = {}
    data_numpy_list['train'] = [x for x in glob.glob(os.path.join(data_dir, 'train/**/*.npy'), recursive=True)]
    data_numpy_list['val'] = [x for x in glob.glob(os.path.join(data_dir, 'val/**/*.npy'), recursive=True)]
    print(data_numpy_list)
    # ImageFolder is a PyTorch class - it expects <class1-name>, <class2-name>, ...folders under the root path you give it
    datasets = {x: NumpyDataset(data_numpy_list[x], data_transforms) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}
    criterion = nn.CrossEntropyLoss()


    custom_cnn = CustomCNN().to(device)
    optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, custom_cnn.parameters()))

    print(f"number of params in model {count_parameters(custom_cnn)}")
    model_conv, metrics = train_model(custom_cnn, criterion, optimizer_conv, num_epochs=3000, path=path1)

    train_loss = np.array(metrics['train_loss'])
    val_loss = np.array(metrics['val_loss'])
    train_acc = np.array(metrics['train_acc'])
    val_acc = np.array(metrics['val_acc'])

    import csv

    np.savetxt('train_loss.csv', train_loss, delimiter=',')
    np.savetxt('val_loss.csv', val_loss, delimiter=',')
    np.savetxt('train_acc.csv', train_acc, delimiter=',')
    np.savetxt('val_acc.csv', val_acc, delimiter=',')

    podaci_za_test = [x for x in glob.glob(os.path.join(data_dir, 'test/**/*.npy'), recursive=True)]
    datasets1 = {x: NumpyDataset(podaci_za_test, data_transforms) for x in ['test']}
    podaci = {x: torch.utils.data.DataLoader(datasets1[x], batch_size=BATCH_SIZE, shuffle=True) for x in
              ['test']}


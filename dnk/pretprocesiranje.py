import numpy as np
import csv
import random
from PIL import Image
import os

def ucitavanje_podataka(putanja1,putanja2):
    podaci1 = csv.reader(open(putanja1, encoding='utf-8'))
    podaci2 = csv.reader(open(putanja2, encoding='utf-8'))
    podaci = []

    for a in podaci1:
        podaci.append([a[0],a[1],a[2]])
    for b in podaci2:
        podaci.append([b[0],b[1],b[2]])

    random.seed(2)
    random.shuffle(podaci)
    return podaci

def vektorizacija(podaci):
    najveca_duzina = 0
    for a in podaci:
        if len(a[1]) > najveca_duzina:
            najveca_duzina = len(a[1])

    # padding with "N" to max_seq_len
    for a in podaci:
        a[1] += "N" * (najveca_duzina - len(a[1]))

    # tranformation of data set:one_hot encoding
    prevodjenje_sekvence = {"A": [[1], [0], [0], [0]], "U": [[0], [1], [0], [0]], \
              "T": [[0], [1], [0], [0]], "G": [[0], [0], [1], [0]], \
              "C": [[0], [0], [0], [1]], "N": [[0], [0], [0], [0]]}
    prevodjenje_labela = {"TRUE": [1], "FALSE": [0]}  # TRUE:Mirtrons  FALSE:canonical microRN

    vektorizovani_podaci = []
    for a in podaci:
        niz = []
        for slovo in a[1]:
            niz.append(prevodjenje_sekvence[slovo])

        vektorizovani_podaci.append([a[0], niz, prevodjenje_labela[a[2]]])

    return vektorizovani_podaci

def podela_podataka(data_vectors):
    X_train, y_train, X_test, y_test = [],[],[],[]
    i,j = 0,0
    for item in data_vectors:
        if item[2]==[1]:
            i = i + 1
            if i <= 201:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                    X_train.append(item[1])
                    y_train.append(item[2])
        else:
            j = j + 1
            if  j<= 200:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                X_train.append(item[1])
                y_train.append(item[2])

    return X_train, y_train, X_test, y_test

putanja1 = "Data/miRBase_set.csv"
putanja2 = "Data/putative_mirtrons_set.csv"

#niz od podataka za svaku sekvencu (podatak je niz od id-ja, sekvence i labele)
podaci = ucitavanje_podataka(putanja1,putanja2)

vektorizovani_podaci = vektorizacija(podaci)
X_train, y_train, X_test, y_test = podela_podataka(vektorizovani_podaci)
X_train = np.array(X_train).transpose(0, 3, 1, 2)
y_train = np.array(y_train)
X_test = np.array(X_test).transpose(0, 3, 1, 2)
y_test = np.array(y_test)

savеpath = 'Data/'
for i in range(len(X_train)):
    np.save(savеpath + 'train/' + str(y_train[i][0]) + '/' + str(i) + '.npy', X_train[i])

for i in range(len(X_test)):
    with open(savеpath + 'val/' + str(y_test[i][0]) + '/' + str(i) + '.npy', 'wb') as f:
        np.save(f, X_test[i])

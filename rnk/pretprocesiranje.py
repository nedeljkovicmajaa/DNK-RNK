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

    random.shuffle(data_vectors)
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

X_test2 = X_test[:200]
y_test2 = y_test[:200]

X_test1 = X_test[200:]
y_test1 = y_test[200:]

sav??path = 'Data/'

for i in range(len(X_train)):
    np.save(sav??path + 'train/' + str(y_train[i][0]) + '/' + str(i) + '.npy', X_train[i])

for i in range(len(X_test1)):
    with open(sav??path + 'val/' + str(y_test1[i][0]) + '/' + str(i) + '.npy', 'wb') as f:
        np.save(f, X_test1[i])

for i in range(len(X_test2)):
    with open(sav??path + 'test/' + str(y_test2[i][0]) + '/' + str(i) + '.npy', 'wb') as f:
        np.save(f, X_test2[i])

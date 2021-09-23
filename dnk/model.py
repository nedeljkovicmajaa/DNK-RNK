import pretprocesiranje
import numpy as np

LR = 0.001  # learning rate
TRAINING_ITER = 10000  # iteration times
BATCH_SIZE = 18  # batch size of input

SEQUENCE_LENGTH = 164  # sequence length of input
EMBEDDING_SIZE = 4  # char embedding size(sequence width of input)
# CONV_SIZE = 3    #first filter size
# CONV_DEEP = 128   #number of first filter(convolution deepth)

STRIDES = [1, 1, 1, 1]  # the strid in each of four dimensions during convolution
KSIZE = [1, 164, 1, 1]  # pooling window size

FC_SIZE = 128  # nodes of full-connection layer
NUM_CLASSES = 2  # classification number

DROPOUT_KEEP_PROB = 0.5  # keep probability of dropout

putanja1 = "Data/miRBase_set.csv"
putanja2 = "Data/putative_mirtrons_set.csv"

#niz od podataka za svaku sekvencu (podatak je niz od id-ja, sekvence i labele)
podaci = pretprocesiranje.ucitavanje_podataka(putanja1,putanja2)

vektorizovani_podaci = pretprocesiranje.vektorizacija(podaci)
X_train, y_train, X_test, y_test = pretprocesiranje.podela_podataka(vektorizovani_podaci)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

dataset_size = len(X_train)
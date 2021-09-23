import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pt

train_loss = np.loadtxt('train_loss.csv', delimiter=',')
val_loss = np.loadtxt('val_loss.csv', delimiter=',')
train_acc = np.loadtxt('train_acc.csv', delimiter=',')
val_acc = np.loadtxt('val_acc.csv', delimiter=',')

x = np.linspace(1, 20, 20)

# plotting
pt.title("Train loss")
pt.xlabel("epoch")
pt.ylabel("loss")
pt.plot(x, train_loss, color="red")
pt.show()